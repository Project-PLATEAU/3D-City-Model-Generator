import torch.nn.functional as nnf
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Dict
import torch
import random
from torch import nn, Tensor
from typing import Tuple
from einops import rearrange, repeat, reduce

import logging

from typing import List

from .mesh_opt import meshOPTForCasualLM

from collections import OrderedDict

from classifier_free_guidance_pytorch import TextEmbeddingReturner

def discretize(
        t: Tensor,
        continuous_range: Tuple[float, float],
        num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo
    t = (t - lo) / (hi - lo)  # cube normalize
    t *= num_discrete
    t -= 0.5
    return t.round().long().clamp(min=0, max=num_discrete - 1)


def undiscretize(
        t: Tensor,
        continuous_range=Tuple[float, float],
        num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo
    t = t.float()
    t += 0.5
    t /= num_discrete  # cube normalize
    return t * (hi - lo) + lo


def train_aug(vertices, faces):
    # print(vertices.shape, faces.shape)
    
    v_mask = vertices == -1
    f_mask = faces == -1

    # scale
    # min_scale = 0.75
    # max_scale = 1.
    # scale_factor = torch.rand((vertices.shape[0], 3), device='cuda') * (max_scale - min_scale) + min_scale
    # # possible_values = np.arange(0.75, 1.0, 0.005)
    # # scale_factor = np.random.choice(possible_values)
    # vertices = vertices * scale_factor.unsqueeze(1)

    # # translation
    # min_val = -0.02
    # max_val = 0.02
    # translation = torch.rand((vertices.shape[0], 3), device='cuda') * (max_val - min_val) + min_val
    # vertices = vertices + translation.unsqueeze(1)

    # # rotation
    # r = torch.tensor(random.uniform(0., 360.0) / 180 * torch.pi)
    # rotation_matrix = torch.tensor([
    #     [torch.cos(r), 0, torch.sin(r)],
    #     [0, 1, 0],
    #     [-torch.sin(r), 0, torch.cos(r)]
    # ]).cuda()
    # vertices = vertices @ rotation_matrix.T

    # to ground
    vertices[v_mask] = 100.
    min_y = vertices[:, :, 1].min(dim=1).values
    difference = -0.95 - min_y
    vertices[:, :, 1] += difference.unsqueeze(1)
    
    # extract footprint
    faces[f_mask] = 10000
    v_count = (vertices[:, :, 1] < -0.94).sum(dim=1)
    f_count = (faces[:, :, -1] < v_count.view(-1, 1)).sum(dim=1)

    f_count_max = torch.max(f_count)
    faces_footprint = faces.clone()[:, :f_count_max]
    faces_footprint_mask = torch.arange(f_count_max).cuda().unsqueeze(0) >= f_count.unsqueeze(1)
    faces_footprint[faces_footprint_mask] = -1

    faces[f_mask] = -1
    vertices[v_mask] = -1.

    # res=trimesh.Trimesh(vertices=vertices[5].cpu().numpy(), faces=faces_footprint[5].cpu().numpy())
    # ori=trimesh.Trimesh(vertices=vertices[5].cpu().numpy(), faces=faces[5].cpu().numpy())
    # res = trimesh.util.concatenate([res, ori.apply_translation([2, 0, 0])])
    # res.export('/fast/zcb/code/cbzhao/meshgpt-pytorch/plateau_lod2_type5/test2/type5_11.obj')
    # aaa

    return vertices, faces_footprint, f_count


class MeshTokenizer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.pad_id = -1
        self.num_discrete_coors = args.n_discrete_size  # default: 800
        self.codebook_size = args.n_discrete_size  # default: 128
        self.coor_continuous_range = (-1., 1.)

    def tokenize(self, data_dict: dict) -> dict:
        '''
        Turn 3D meshes into sequential tokens: <bos> [<x>, <y>, <z>], ... <eos>
        '''

        ### 3D mesh face parsing
        vertices = data_dict['vertices']  # batch x nv x 3
        faces = data_dict['faces']  # batch x nf x 3
        face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')  # batch x nf

        batch, num_vertices, num_coors = vertices.shape
        _, num_faces, _ = faces.shape

        # fill padding tokens with 0, to prevent gather idx error
        face_without_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0)

        # collect vertice coordinates per-face: b x nf x nv x c
        faces_vertices = repeat(face_without_pad, 'b nf nv -> b nf nv c', c=num_coors)
        vertices = repeat(vertices, 'b nv c -> b nf nv c', nf=num_faces)
        face_coords = vertices.gather(-2, faces_vertices.long())

        # continuous to discrete face coords: b x nf x nv x c
        discrete_face_coords = discretize(
            face_coords,
            continuous_range=self.coor_continuous_range,
            num_discrete=self.num_discrete_coors
        )

        # pad invalid faces with <pad_id>: batch x nf x nv x c
        discrete_padded_coords = discrete_face_coords.masked_fill(
            ~rearrange(face_mask, 'b nf -> b nf 1 1'),
            self.pad_id
        )

        ### mesh to sequence convertion: batch x ntokens
        input_ids = discrete_padded_coords.reshape(batch, -1)
        attention_mask = (input_ids != self.pad_id).float()
        # reserve two spots:
        #     input_ids: <bos> ... <eos> <pad> ... => <pad> ... <pad> <pad> ...
        #     attn_mask:    1  ...    1     0  ... =>    1  ...    1     0  ...
        place_holder = torch.ones_like(input_ids[:, [0]])  # batch x 1
        input_ids = torch.cat((place_holder * self.pad_id, input_ids, place_holder * self.pad_id), dim=1)
        attention_mask = torch.cat((place_holder, place_holder, attention_mask), dim=1)

        ### meshXL inputs
        data_dict['input_ids'] = input_ids.long()  # batch x (nf * 3 * 3 + 2)
        data_dict['attention_mask'] = attention_mask.float()  # batch x (nf * 3 * 3 + 2)

        # discard <bos> and <eos> tokens
        data_dict['codes'] = discrete_padded_coords.long()  # batch x (nf * 3 * 3)
        data_dict['discrete_face_coords'] = discrete_face_coords

        return data_dict

    def detokenize(self, input_ids: Tensor) -> dict:
        '''
        Turn sequential tokens: <bos> [<x>, <y>, <z>], ... <eos> into 3D meshes
        '''
        # input_ids: b (n q) or b n q, without <bos> or <eos>
        input_ids = input_ids.reshape(input_ids.shape[0], -1)
        # batch x nface
        face_mask = reduce(
            input_ids != self.pad_id, 'b (nf c) -> b nf', 'all', c=9
        )

        # batch x (nface x 9) -> batch x nface x 3 x 3
        pred_face_coords = input_ids.reshape(input_ids.shape[0], -1, 9)
        pred_face_coords = rearrange(
            pred_face_coords, '... (v c) -> ... v c', v=3
        )

        # back to continuous space
        continuous_coors = undiscretize(
            pred_face_coords,
            num_discrete=self.num_discrete_coors,
            continuous_range=self.coor_continuous_range
        )
        # mask padding coordinates out with nan
        continuous_coors = continuous_coors.masked_fill(
            ~rearrange(face_mask, 'b nf -> b nf 1 1'),
            float('nan')
        )
        output_dict = {}
        output_dict['recon_faces'] = continuous_coors

        return output_dict

    def forward(self, data_dict: dict) -> dict:
        encoder_output = self.tokenize(data_dict)
        decoder_output = self.detokenize(
            input_ids=encoder_output['codes'],
        )
        data_dict.update(encoder_output)
        data_dict.update(decoder_output)
        return data_dict


class MeshXL(nn.Module):

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def __init__(self, args):
        super().__init__()

        self.tokenizer = MeshTokenizer(args)

        # causal LM model initialization
        self.vocab_size = self.tokenizer.codebook_size + 3
        self.bos_token_id = self.tokenizer.codebook_size
        self.eos_token_id = self.tokenizer.codebook_size + 1
        self.pad_token_id = self.tokenizer.codebook_size + 2

        config = AutoConfig.from_pretrained(
            args.llm,
            n_positions=8192,
            max_position_embeddings=8192,
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id
        )
        
        self.add_condition = config.add_cross_attn
        self.conditioner = None
        self.dim_condition = None
        
        text_condition_model_types = 't5'
        text_condition_model_kwargs = (dict(), )
        text_condition_cond_drop_prob = 0.0
        
        if self.add_condition:
            self.conditioner = TextEmbeddingReturner(
                model_types = text_condition_model_types,
                model_kwargs = text_condition_model_kwargs,
                cond_drop_prob = text_condition_cond_drop_prob,
                text_embed_pad_value = -1.
            )
            
            self.dim_condition = self.conditioner.dim_latent
        
        # print(config)
        # aa

        config.word_embed_proj_dim = config.hidden_size

        if not self.add_condition:
            self.transformer = AutoModelForCausalLM.from_pretrained(
                args.llm,
                config=config,
                ignore_mismatched_sizes=True
            )
        else:
            self.transformer = meshOPTForCasualLM(config = config)
            # self.transformer = load_pretrained_opt(self.transformer, 
            #                                        "Building_Generation_Opening/BldgXL/config/mesh-xl-350m/pytorch_model.bin", 
            #                                        device = 'cuda')
        # print(self.transformer)
        
        self.transformer.to_bettertransformer()

        # setting status for all parameters
        self.train()

    @torch.no_grad()
    def embed_texts(self, texts):
        single_text = not isinstance(texts, list)
        if single_text:
            texts = [texts]

        assert self.conditioner is not None
        text_embeds = self.conditioner.embed_texts(texts).detach()

        if single_text:
            text_embeds = text_embeds[0]

        return text_embeds

    def forward(
            self,
            data_dict: dict = None,
            is_eval: bool = False,
            is_generate: bool = False,
            num_return_sequences: int = 8,
            generation_config: Dict = dict(
                do_sample=True,
                top_k=50,
                top_p=0.95,
                # no_repeat_ngram_size=9,
            )
    ) -> dict:

        if not is_eval:
            return self.train_one_step(data_dict)

        if is_eval and not is_generate:
            return self.perplexity(data_dict)

        if is_eval and is_generate:
            return self.generate(
                data_dict=data_dict,
                num_return_sequences=num_return_sequences,
                generation_config=generation_config
            )

        raise NotImplementedError('training status undefined!')
        return

    def loss_wrapper(self, loss: Tensor) -> Tensor:
        # parameter activation: it is a l2 loss with 0 weight
        for param in self.parameters():
            loss += 0 * torch.sum(param ** 2)
        return loss

    def train_one_step(self, data_dict: dict) -> dict:
        data_dict['vertices'], _, _ = train_aug(data_dict['vertices'].clone(), data_dict['faces'].clone())
        data_dict = self.tokenizer.tokenize(data_dict)

        input_ids = data_dict['input_ids']  # batch x ntoken
        attention_mask = data_dict['attention_mask']  # batch x ntoken

        # print(data_dict['attention_mask'])
        # aa

        # parse input with <bos> and <eos> tokens
        input_ids[input_ids == self.tokenizer.pad_id] = self.pad_token_id  # <pad> xxx <pad> <pad>
        input_ids[:, 0] = self.bos_token_id  # <bos> xxx <pad> <pad>
        eos_pos_id = attention_mask.sum(1, keepdim=True) - 1
        input_ids = torch.scatter(  # <bos> xxx <eos> <pad>
            input_ids,
            1,
            eos_pos_id.long(),
            torch.ones_like(input_ids) * self.eos_token_id
        )

        target = input_ids.clone()
        target[attention_mask == 0] = -100  # not loss for the padding tokens

        if self.add_condition:
            text_embeds = self.embed_texts(data_dict['texts'])
            # print(input_ids)
            # print(text_embeds)
            # aa
            output = self.transformer(
                input_ids=input_ids.long(),
                key_value_states=text_embeds
            )
        else:
            output = self.transformer(
                input_ids=input_ids.long(),
            )

        # Forward padd, calling causal llm with better transformer.
        

        # compute loss with shift one-token right
        logit = output.logits[:, :-1]  # batch x ntoken x vocab
        label = target[:, 1:]  # batch x ntoken

        final_loss = nnf.cross_entropy(
            logit.permute(0, 2, 1),  # batch x vocab x ntoken
            label,
        )  # batch x ntoken

        data_dict['loss'] = self.loss_wrapper(final_loss)
        data_dict['gen_loss'] = final_loss

        return data_dict['gen_loss']

    @torch.no_grad()
    def perplexity(self, data_dict: dict) -> dict:

        data_dict = self.tokenizer.tokenize(data_dict)

        input_ids = data_dict['input_ids']  # batch x ntoken
        attention_mask = data_dict['attention_mask']  # batch x ntoken

        # set pad_token_id = eos_token_id
        input_ids[input_ids == self.tokenizer.pad_id] = self.pad_token_id  # <pad> xxx <pad> <pad>
        input_ids[:, 0] = self.bos_token_id  # <sos> xxx <pad> <pad>
        eos_pos_id = attention_mask.sum(1, keepdim=True) - 1
        input_ids = torch.scatter(  # <bos> xxx <eos> <pad>
            input_ids,
            1,
            eos_pos_id.long(),
            torch.ones_like(input_ids) * self.eos_token_id
        )

        # llm loss calculation
        output = self.transformer(
            input_ids=input_ids.long(),
        )

        # compute loss with shift token right
        logit = output.logits[:, :-1]  # batch x (ntoken - 1) x vocab
        label = input_ids[:, 1:]  # batch x (ntoken - 1)
        masks = attention_mask[:, 1:]  # batch x (ntoken - 1)
        loss_per_token = nnf.cross_entropy(
            logit.permute(0, 2, 1),  # batch x (ntoken - 1) x ntoken
            label,  # batch x (ntoken - 1)
            reduction='none'
        )  # batch x ntoken

        # compute negative log likelihood for each sequence
        neg_log_likelihood = torch.sum(loss_per_token * masks, dim=1) / torch.sum(masks, dim=1)

        data_dict['neg_log_likelihood'] = neg_log_likelihood  # batch,
        return data_dict

    @torch.no_grad()
    def generate(self, data_dict: dict = None, num_return_sequences: int = 8, generation_config: dict = dict()) -> dict:

        net_device = next(self.parameters()).device
        max_length = 8192
        output_ids = torch.ones(num_return_sequences, max_length).long().to(net_device) * self.eos_token_id

        # batch x ntokens
        results = self.transformer.generate(
            max_new_tokens=max_length - 1,
            num_return_sequences=num_return_sequences,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.eos_token_id,
            **generation_config
        )
        output_ids[:, :results.shape[1]] = results

        # discard <bos> and <eos> tokens to pad tokens
        output_ids = output_ids[:, 1: -1]
        output_ids[output_ids == self.eos_token_id] = self.tokenizer.pad_id

        decoder_output = self.tokenizer.detokenize(input_ids=output_ids)

        return decoder_output

    @torch.no_grad()
    def generate_partial(self, data_dict: dict = None, n_samples: int = 8) -> dict:
        data_dict = self.tokenizer.tokenize(data_dict)
        input_ids = data_dict['input_ids']  # 1 x ntoken
        attention_mask = data_dict['attention_mask']  # 1 x ntoken

        # replace padding tokens
        input_ids[:, 0] = self.bos_token_id  # <sos> xxx <pad> <pad>
        eos_pos_id = attention_mask.sum(1, keepdim=True) - 1
        input_ids = torch.scatter(
            input_ids,
            1,
            eos_pos_id.long(),
            torch.ones_like(input_ids) * self.eos_token_id
        )
        
        # print(self.transformer)
        
        # embed texts
        condition = None
        if self.add_condition:
            text_embeds = self.embed_texts(data_dict['texts'])
            condition = text_embeds
            # print(condition, condition.shape)

        # conditioned on 1/4 the shape
        input_ids = input_ids[:, attention_mask[0] == 1]  # 1 x [<bos> ... <eos>]
        num_faces = (input_ids.shape[1] - 2) // 9
        kept_length = (num_faces // 1) * 9 + 1
        input_ids = input_ids[:, :kept_length]  # 1 x [<bos> ...]

        net_device = next(self.parameters()).device
        max_length = 7202
        outputs = torch.ones(n_samples, max_length).long().to(net_device) * self.eos_token_id
        # batch x ntokens
        results = self.transformer.generate(
            input_ids=input_ids,
            key_value_states=condition, 
            max_new_tokens=max_length - input_ids.shape[1],
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=n_samples,
            # num_beams=n_samples,
            # no_repeat_ngram_size=9,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.eos_token_id,
        )
        outputs[:, :results.shape[1]] = results
        # batch x ntokens ====> batch x ntokens x D
        outputs = outputs[:, 1: -1]
        outputs[outputs == self.eos_token_id] = self.tokenizer.pad_id
        decoder_output = self.tokenizer.detokenize(outputs)

        condition_output = self.tokenizer.detokenize(input_ids[:, 1:])
        decoder_output['partial_mesh'] = condition_output['recon_faces']

        return decoder_output


def load_pretrained_opt(model: meshOPTForCasualLM, 
                        pretrained_path: str, 
                        device: str = 'cuda'):
    available_weights = OrderedDict()
    
    checkpoint = torch.load(pretrained_path, map_location = 'cuda')
    
    pretrained_ext = pretrained_path[-4:]
    print(pretrained_ext)
    
    if pretrained_ext == '.pth':
        checkpoint = checkpoint['model']
    # print(checkpoint.keys())
    
    # pretrained_opt = MeshXL.load_state_dict(checkpoint)
    # pretrained_dict = pretrained_opt.state_dict()
    
    # for name, param in checkpoint.items():
    #     print(name)
    # aaa
    
    model_dict = model.state_dict()
    
    # excluded_name = ['transformer.model.decoder.embed_tokens.weight', 
    #                  'transformer.lm_head.weight']
    
    prefix = 'model' if pretrained_ext == '.bin' else 'transformer'
    for name, param in model_dict.items():
        # matched_name = name
        if pretrained_ext == '.bin':
            matched_name = name[name.index('.') + 1:]
        else:
            matched_name = prefix + '.' + name

        if matched_name in checkpoint:
            if checkpoint[matched_name].shape == param.shape:
                if pretrained_ext == '.bin':
                    saved_name = prefix + '.' + matched_name
                else:
                    saved_name = matched_name[len(prefix) + 1:]
                
                available_weights[saved_name] = checkpoint[matched_name]
            else:
                logging.warning(f"shape mismatched for {name}: "
                                f'pretrained {checkpoint[name].shape} vs '
                                f'modified {param.shape}')
        else:
            logging.warning(f"name not found: {matched_name}")
    
    model_dict.update(available_weights)
    model.load_state_dict(model_dict)

    print(pretrained_ext)
    print('ckpt loaded. \n')
    
    return model.to(device)


def get_model(args):
    model = MeshXL(args)
    return model
