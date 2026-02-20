from __future__ import annotations

from pathlib import Path
from functools import partial
from packaging import version
from contextlib import nullcontext

import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from pytorch_custom_utils import (
    get_adam_optimizer,
    OptimizerWithWarmupSchedule
)

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs


from beartype.typing import Tuple, Type, List

import matplotlib.pyplot as plt
from tqdm import tqdm

from models import MeshXL
from torch.nn.utils.rnn import pad_sequence
from torch import is_tensor
from beartype import beartype
from beartype.door import is_bearable
from jaxtyping import jaxtyped
from environs import Env


# constants

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters = True
)

def always(value):
    def inner(*args, **kwargs):
        return value
    return inner

def identity(t):
    return t
env = Env()
env.read_env()
should_typecheck = env.bool('TYPECHECK', False)
typecheck = jaxtyped(typechecker = beartype) if should_typecheck else identity
beartype_isinstance = is_bearable if should_typecheck else always(True)


def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def cycle(dl):
    while True:
        for data in dl:
            yield data

def maybe_del(d: dict, *keys):
    for key in keys:
        if key not in d:
            continue

        del d[key]

def first(it):
    return it[0]

def custom_collate(data, pad_id = -1):
    is_dict = isinstance(first(data), dict)

    if is_dict:
        keys = first(data).keys()
        data = [d.values() for d in data]

    output = []

    for datum in zip(*data):
        if is_tensor(first(datum)):
            datum = pad_sequence(datum, batch_first = True, padding_value = pad_id)
        else:
            datum = list(datum)

        output.append(datum)

    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))

    return output


class MeshxlTrainer(Module):
    @typecheck
    def __init__(
            self,
            model: MeshXL,
            dataset: Dataset,
            num_train_steps: int,
            batch_size: int,
            grad_accum_every: int,
            learning_rate: float = 2e-4,
            weight_decay: float = 0.,
            max_grad_norm: float | None = 0.5,
            scheduler: Type[_LRScheduler] | None = None,
            scheduler_kwargs: dict = dict(),
            accelerator_kwargs: dict = dict(),
            optimizer_kwargs: dict = dict(),

            checkpoint_every=1000,
            checkpoint_every_epoch: Type[int] | None = None,
            checkpoint_folder='./checkpoints',
            data_kwargs: Tuple[str, ...] = ('vertices', 'faces', 'face_edges', 'text'),
            warmup_steps=1000,
            use_wandb_tracking=False
    ):
        super().__init__()

        # experiment tracker

        self.use_wandb_tracking = use_wandb_tracking

        if use_wandb_tracking:
            accelerator_kwargs['log_with'] = 'wandb'

        if 'kwargs_handlers' not in accelerator_kwargs:
            accelerator_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]

        self.accelerator = Accelerator(**accelerator_kwargs)

        self.model = model

        optimizer = get_adam_optimizer(
            model.parameters(),
            lr=learning_rate,
            wd=weight_decay,
            filter_by_requires_grad=True,
            **optimizer_kwargs
        )

        self.optimizer = OptimizerWithWarmupSchedule(
            accelerator=self.accelerator,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm
        )

        self.dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size, 
            drop_last=True,
            collate_fn=partial(custom_collate, pad_id=model.pad_id)
        )


        if hasattr(dataset, 'data_kwargs') and exists(dataset.data_kwargs):
            assert beartype_isinstance(dataset.data_kwargs, List[str])
            self.data_kwargs = dataset.data_kwargs
        else:
            self.data_kwargs = data_kwargs

        (
            self.model,
            self.dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.dataloader
        )

        self.grad_accum_every = grad_accum_every
        self.num_train_steps = num_train_steps
        self.register_buffer('step', torch.tensor(0))

        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok=True, parents=True)

    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step=self.step.item())

    @property
    def device(self):
        return self.unwrapped_model.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)

    def next_data_to_forward_kwargs(self, dl_iter) -> dict:
        data = next(dl_iter)

        if isinstance(data, tuple):
            forward_kwargs = dict(zip(self.data_kwargs, data))

        elif isinstance(data, dict):
            forward_kwargs = data
            
        # Move data to device if not already there
        target_device = next(self.model.parameters()).device
        for key, value in forward_kwargs.items():
            if hasattr(value, 'to') and hasattr(value, 'device'):
                if value.device != target_device:
                    forward_kwargs[key] = value.to(target_device, non_blocking=True)

        return forward_kwargs

    def save(self, path, overwrite=True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(
            model=self.unwrapped_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            step=self.step.item(),
        )

        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path))

        self.model.load_state_dict(pkg['model'])
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.step.copy_(pkg['step'])

    def forward(self):
        step = self.step.item()
        dl_iter = cycle(self.dataloader)

        if self.should_validate:
            val_dl_iter = cycle(self.val_dataloader)

        while step < self.num_train_steps:

            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                forward_kwargs = self.next_data_to_forward_kwargs(dl_iter)

                with self.accelerator.autocast(), maybe_no_sync():
                    loss = self.model(**forward_kwargs)

                    self.accelerator.backward(loss / self.grad_accum_every)

            self.print(f'loss: {loss.item():.3f}')

            self.log(loss=loss.item())

            self.optimizer.step()
            self.optimizer.zero_grad()
            # Use set_to_none on underlying optimizer for better memory efficiency
            if hasattr(self.optimizer, 'optimizer'):
                self.optimizer.optimizer.zero_grad(set_to_none=True)

            step += 1
            self.step.add_(1)

            self.wait()
            
            # Clear cache every 50 steps to prevent memory accumulation
            if step % 50 == 0:
                torch.cuda.empty_cache()

            if self.is_main and self.should_validate and divisible_by(step, self.val_every):

                total_val_loss = 0.
                self.unwrapped_model.eval()

                num_val_batches = self.val_num_batches * self.grad_accum_every

                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():
                        forward_kwargs = self.next_data_to_forward_kwargs(val_dl_iter)

                        val_loss = self.unwrapped_model(**forward_kwargs)

                        total_val_loss += (val_loss / num_val_batches)

                self.print(f'valid recon loss: {total_val_loss:.3f}')

                self.log(val_loss=total_val_loss)

            self.wait()

            if self.is_main and divisible_by(step, self.checkpoint_every):
                checkpoint_num = step // self.checkpoint_every
                self.save(self.checkpoint_folder / f'mesh-transformer.ckpt.{checkpoint_num}.pt')

            self.wait()

        self.print('training complete')

    def train(self, num_epochs, stop_at_loss=None, diplay_graph=False):
        epoch_losses = []
        epoch_size = len(self.dataloader)
        self.model.train()

        for epoch in range(num_epochs):
            total_epoch_loss = 0.0

            progress_bar = tqdm(enumerate(self.dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}',
                                total=len(self.dataloader))
            for batch_idx, batch in progress_bar:
                # Move batch to device if needed
                if isinstance(batch, dict):
                    target_device = next(self.model.parameters()).device
                    for key, value in batch.items():
                        if hasattr(value, 'to') and hasattr(value, 'device'):
                            if value.device != target_device:
                                batch[key] = value.to(target_device, non_blocking=True)

                is_last = (batch_idx + 1) % self.grad_accum_every == 0
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                with self.accelerator.autocast(), maybe_no_sync():
                    total_loss = self.model(batch)
                    self.accelerator.backward(total_loss / self.grad_accum_every)

                current_loss = total_loss.item()
                total_epoch_loss += current_loss

                progress_bar.set_postfix(loss=current_loss)

                if is_last or (batch_idx + 1 == len(self.dataloader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # Use set_to_none on underlying optimizer for better memory efficiency
                    if hasattr(self.optimizer, 'optimizer'):
                        self.optimizer.optimizer.zero_grad(set_to_none=True)
                    
                # Clear cache periodically to prevent memory accumulation
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()

            avg_epoch_loss = total_epoch_loss / epoch_size
            epochOut = f'Epoch {epoch + 1} average loss: {avg_epoch_loss}, Learning Rate = {self.optimizer.optimizer.param_groups[0]["lr"]}'

            epoch_losses.append(avg_epoch_loss)

            if len(epoch_losses) >= 4 and avg_epoch_loss > 0:
                avg_loss_improvement = sum(epoch_losses[-4:-1]) / 3 - avg_epoch_loss
                epochOut += f'          avg loss speed: {avg_loss_improvement}'
                if avg_loss_improvement > 0 and avg_loss_improvement < 0.2:
                    epochs_until_0_3 = max(0, abs(avg_epoch_loss - 0.3) / avg_loss_improvement)
                    if epochs_until_0_3 > 0:
                        epochOut += f' epochs left: {epochs_until_0_3:.2f}'

            self.wait()
            self.print(epochOut)

            if self.is_main and self.checkpoint_every_epoch is not None and (
                    self.checkpoint_every_epoch == 1 or (epoch != 0 and epoch % self.checkpoint_every_epoch == 0)):
                self.save(
                    self.checkpoint_folder / f'mesh-transformer.ckpt.epoch_{epoch}_avg_loss_{avg_epoch_loss:.3f}.pt')

            if stop_at_loss is not None and avg_epoch_loss < stop_at_loss:
                self.print(f'Stopping training at epoch {epoch} with average loss {avg_epoch_loss}')
                if self.is_main and self.checkpoint_every_epoch is not None:
                    self.save(
                        self.checkpoint_folder / f'mesh-transformer.ckpt.stop_at_loss_avg_loss_{avg_epoch_loss:.3f}.pt')
                break

        self.print('Training complete')
        if diplay_graph:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label='Total Loss')
            plt.title('Training Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.grid(True)
            plt.show()
        return epoch_losses[-1]
