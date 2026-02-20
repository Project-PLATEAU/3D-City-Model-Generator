import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import gc

import torch
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as lr_scheduler
from models import MeshXL
from dataset import MeshDataset
from trainer import MeshxlTrainer

# Enable memory optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)

os.environ['NCCL_P2P_DISABLE'] = "1"
os.environ['NCCL_IB_DISABLE'] = "1"

# dataset_path = '../BldgXL/plateau_lod2_withimg_buf2/plateau_lod2_withimg_buf2.npz'
# dataset = MeshDataset.load(dataset_path)

args_dict = {
    "n_discrete_size": 128, 
    "llm": 'config/mesh-xl-125m'
}

def get_model(args):
    model = MeshXL(args)
    return model

args = argparse.Namespace(**args_dict)
model = get_model(args)

# Enable gradient checkpointing to save memory
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

# total_params = sum(p.numel() for p in model.transformer.model.decoder.layers.parameters()) 
# total_params = f"{total_params / 1000000:.1f}M"
# print(f"Total parameters: {total_params}")

# print(model.transformer)
print(model)
print(type(model.transformer.model.decoder.layers[0].self_attn))
print(type(model.transformer.model.decoder.layers[0].cross_attn))
aaa

intermediate_load = False
if intermediate_load:
    checkpoint = torch.load('checkpoints/mesh-transformer.ckpt.epoch_140_avg_loss_0.081.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    del checkpoint  # Free memory
    torch.cuda.empty_cache()

batch_size = 2 # Max 64
grad_accum_every = 16

learning_rate = 1e-4
max_ep = 500
scheduler = lr_scheduler.CosineAnnealingLR
scheduler_kwargs = {"T_max": max_ep * int(len(dataset.data) / batch_size / grad_accum_every), "eta_min": 1e-6}

trainer = MeshxlTrainer(model = model, warmup_steps = 0, num_train_steps = 100, dataset = dataset,
                        grad_accum_every = grad_accum_every,
                        learning_rate = learning_rate,
                        batch_size = batch_size,
                        checkpoint_every_epoch = 10,
                        scheduler = scheduler,
                        scheduler_kwargs = scheduler_kwargs,
                        # checkpoint_folder=f'{working_dir}'
                        # FP16 training, it doesn't speed up very much but can increase the batch size which will in turn speed up the training.
                        # However it might cause nan after a while.
                        # accelerator_kwargs = {"mixed_precision" : "fp16"}, optimizer_kwargs = { "eps": 1e-7}
                        )

# Enable mixed precision training for memory efficiency
# trainer.accelerator.fp16 = True

loss = trainer.train(max_ep, stop_at_loss = 0.005)

# Clean up after training
torch.cuda.empty_cache()
gc.collect()

# trainer.save('plateau_lod2_type_mixed/plateau_lod2_type_mixed.pt')