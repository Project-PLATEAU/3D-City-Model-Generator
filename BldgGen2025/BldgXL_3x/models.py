from pathlib import Path

import torch.nn.functional as nnf
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Dict
import torch
import random
from torch import nn, Tensor
from typing import Tuple
from einops import rearrange, repeat, reduce
import trimesh
import numpy as np

import logging

from typing import List

from .mesh_opt import meshOPTForCasualLM

from collections import OrderedDict

from classifier_free_guidance_pytorch import TextEmbeddingReturner

import clip
from PIL import Image

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


def train_aug(vertices, faces, img, scale_factor):
    # print(vertices.shape, faces.shape)
    padding_value = -1

    b, _, _ = vertices.shape
    # print(faces, faces.shape)

    degree = random.uniform(0., 360.0)
    r = degree / 180 * torch.pi
    cos_r = torch.cos(torch.tensor(r, device=vertices.device))
    sin_r = torch.sin(torch.tensor(r, device=vertices.device))
    rotation_matrix = torch.tensor([
        [cos_r, 0, sin_r],
        [0, 1, 0],
        [-sin_r, 0, cos_r]
    ], device=vertices.device)
    
    img_degree = degree
    
    faces_footprint_count = torch.zeros(b, dtype=torch.long)
    real_vertex_mask = ~torch.all(vertices == padding_value, dim=-1)
    real_face_mask = ~torch.all(faces == padding_value, dim=-1)
    
    for pos in range(b):
        current_mask = real_vertex_mask[pos]
        
        # mask out padding position
        real_vertices = vertices[pos][current_mask]
        
        # real_vertices_output = real_vertices.detach().cpu().numpy()
        # faces_output = faces[pos].detach().cpu().numpy()
        # mesh_original = trimesh.Trimesh(real_vertices_output, faces_output)
        # mesh_original.export(f'mesh_noaug_pos{pos}.obj')
        
        real_vertices = real_vertices * scale_factor[pos]
        
        # rotation aug
        real_vertices = real_vertices @ rotation_matrix.T
        
        # reverse
        centroid = real_vertices.mean(dim=0)
        centered = real_vertices - centroid
        
        max_abs = centered.abs().max()
        vertices_rescaled = centered * (0.95 / max_abs)
        
        # put to ground
        height_min = vertices_rescaled[:, 1].min(dim=0).values
        difference = -0.95 - height_min
        vertices_rescaled[:, 1] += difference
        
        # vertices_rescaled_output = vertices_rescaled.detach().cpu().numpy()
        # mesh_original = trimesh.Trimesh(vertices_rescaled_output, faces_output)
        # mesh_original.export(f'mesh_aug_pos{pos}.obj')
        
        # maximum scaling
        ## x-z scaling
        xz_coords = vertices_rescaled[:, [0, 2]]
        y_coords = vertices_rescaled[:, 1:2]
        
        xz_min = torch.min(xz_coords, dim=0)[0]
        xz_max = torch.max(xz_coords, dim=0)[0]
        longest_xz_edge = torch.max(xz_max - xz_min)
        
        if longest_xz_edge > 0:
            xz_scale = (2 * 0.95) / longest_xz_edge
        else:
            xz_scale = 1.0
            
        xz_center = (xz_min + xz_max) / 2
        xz_scaled = (xz_coords - xz_center) * xz_scale
        
        ## y scaling
        y_min = torch.min(y_coords)
        y_max = torch.max(y_coords)
        y_range = y_max - y_min
        
        if y_range > 0:
            y_scale = (2 * 0.95) / y_range
        else:
            y_scale = 1.0
            
        y_center = (y_min + y_max) / 2
        y_scaled = (y_coords - y_center) * y_scale
        
        ## put back
        vertices_rescaled_scaleuped = vertices_rescaled.clone()
        vertices_rescaled_scaleuped[:, [0, 2]] = xz_scaled
        vertices_rescaled_scaleuped[:, 1:2] = y_scaled
        
        # vertices_rescaled_scaleuped_output = vertices_rescaled_scaleuped.detach().cpu().numpy()
        # mesh_original = trimesh.Trimesh(vertices_rescaled_scaleuped_output, faces_output)
        # mesh_original.export(f'mesh_aug_scaled_pos{pos}.obj')
        
        # put back
        vertices[pos][current_mask] = vertices_rescaled_scaleuped

        # align faces
        current_mask_face = real_face_mask[pos]
        valid_faces = faces[pos][current_mask_face]

        # Calculate face heights more efficiently
        face_heights = []
        valid_indices = []
        for fidx, face in enumerate(valid_faces):
            if torch.all(face == padding_value):
                continue
            face_vertices = vertices_rescaled_scaleuped[face]
            centroid_height = torch.mean(face_vertices[:, 1])
            face_heights.append(centroid_height)
            valid_indices.append(fidx)

        if face_heights:
            # Convert to tensor for efficient sorting
            face_heights_tensor = torch.stack(face_heights)
            sorted_indices = torch.argsort(face_heights_tensor)

            # Reorder faces based on sorted indices
            sorted_face_indices = [valid_indices[i] for i in sorted_indices]
            sorted_faces = valid_faces[sorted_face_indices]
            faces[pos][current_mask_face] = sorted_faces

            # Clean up temporary tensors
            del face_heights_tensor, sorted_indices
        
        # faces_scaled_output = faces[pos].detach().cpu().numpy()
        # vertices_rescaled_scaleuped_output = vertices_rescaled_scaleuped.detach().cpu().numpy()
        # mesh_original = trimesh.Trimesh(vertices_rescaled_scaleuped_output, faces_scaled_output)
        # mesh_original.export(f'mesh_aug_allscaled_pos{pos}.obj')

        # img[pos] = img[pos].rotate(img_degree, expand=True)
        # img[pos].save("img_test.jpg")
        
        # aa

    # Clean up intermediate tensors to prevent memory leaks
    del rotation_matrix, real_vertex_mask, real_face_mask
    if 'cos_r' in locals():
        del cos_r, sin_r

    # Explicit garbage collection hint for large tensors
    torch.cuda.empty_cache() if vertices.is_cuda else None

    return vertices, faces, img, faces_footprint_count


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

        # Apply coordinate-specific offsets: x=0, y=+128, z=+256
        coord_offsets = torch.tensor([0, self.num_discrete_coors, 2 * self.num_discrete_coors],
                                     device=discrete_face_coords.device,
                                     dtype=discrete_face_coords.dtype)
        discrete_face_coords = discrete_face_coords + coord_offsets.view(1, 1, 1, 3)

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

        # Remove coordinate-specific offsets: x=0, y=-128, z=-256
        # Use codebook_size (128) not num_discrete_coors (800) for the offset
        coord_offsets = torch.tensor([0, self.codebook_size, 2 * self.codebook_size],
                                     device=pred_face_coords.device,
                                     dtype=pred_face_coords.dtype)
        pred_face_coords = pred_face_coords - coord_offsets.view(1, 1, 1, 3)

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
        # Ensure CLIP encoder is in training mode
        if hasattr(self, 'conditioner') and self.conditioner is not None:
            self.conditioner.train(mode)
        return self

    def __init__(self, args):
        super().__init__()

        self.tokenizer = MeshTokenizer(args)

        # causal LM model initialization
        # vocab_size now includes 3 * codebook_size (x, y, z separately) + 3 special tokens
        self.vocab_size = 3 * self.tokenizer.codebook_size + 3
        self.bos_token_id = 3 * self.tokenizer.codebook_size
        self.eos_token_id = 3 * self.tokenizer.codebook_size + 1
        self.pad_token_id = 3 * self.tokenizer.codebook_size + 2

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
        self.add_image_condition = config.add_visual_cross_attn
        self.conditioner = None
        self.dim_condition = None
        
        text_condition_model_types = 't5'
        text_condition_model_kwargs = (dict(), )
        text_condition_cond_drop_prob = 0.0
        
        if self.add_condition:
            if self.add_image_condition:
                self.conditioner, self.preprocessing = clip.load("ViT-B/32", device='cpu')
                self.conditioner = self.conditioner.visual

                # Enable gradients for CLIP image encoder
                for param in self.conditioner.parameters():
                    param.requires_grad = True

                self.image_projection = nn.Sequential(
                    nn.Linear(config.cond_attn_embed_dim, config.cond_attn_embed_dim),
                    nn.LayerNorm(config.cond_attn_embed_dim)
                )
            else:
                self.conditioner = TextEmbeddingReturner(
                    model_types = text_condition_model_types,
                    model_kwargs = text_condition_model_kwargs,
                    cond_drop_prob = text_condition_cond_drop_prob,
                    text_embed_pad_value = -1.
                )
                
                self.dim_condition = self.conditioner.dim_latent

        self.pad_id = self.tokenizer.pad_id

        if not self.add_condition:
            self.transformer = AutoModelForCausalLM.from_pretrained(
                args.llm,
                config=config,
                ignore_mismatched_sizes=True
            )
        else:
            self.transformer = meshOPTForCasualLM(config = config)
            # self.transformer = load_pretrained_opt(self.transformer, 
            #                                        "config/mesh-xl-125m/pytorch_model.bin", 
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
    
    def embed_images(self, img):
        if self.add_image_condition:
            # During inference, disable gradients for efficiency
            # During training, gradients will flow through
            if not self.training:
                with torch.no_grad():
                    img_features = self.conditioner(img).unsqueeze(1)
            else:
                img_features = self.conditioner(img).unsqueeze(1)

            # print(img_features)
            # aaa
        else:
            raise TypeError(self.conditioner)

        return img_features

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
        img_clone = data_dict['img']
        data_dict['vertices'], data_dict['faces'], data_dict['img'], footprint_count = train_aug(data_dict['vertices'].clone(), 
                                                                                                data_dict['faces'].clone(), 
                                                                                                img_clone, 
                                                                                                data_dict['scale'])
        img_preprocessed = []
        for img in data_dict['img']:
            img_preprocessed.append(self.preprocessing(img).to(data_dict['vertices'].device))
        
        img_preprocessed = torch.stack(img_preprocessed)
        # print(img_preprocessed)
        # aaaaaa
        
        # real_vertices_output = data_dict['vertices'][0].detach().cpu().numpy()
        # faces_output = data_dict['faces'][0].detach().cpu().numpy()
        # mesh_original = trimesh.Trimesh(real_vertices_output, faces_output)
        # mesh_original.export('mesh_aug_returned_pos0.obj')
        # adsl
        data_dict = self.tokenizer.tokenize(data_dict)
        # print(data_dict)
        
        input_ids = data_dict['input_ids']  # batch x ntoken
        attention_mask = data_dict['attention_mask']  # batch x ntoken

        # print(data_dict['input_ids'], data_dict['attention_mask'])
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
        
        # print(input_ids, data_dict['attention_mask'])
        # aa

        target = input_ids.clone()
        target[attention_mask == 0] = -100  # not loss for the padding tokens

        if self.add_condition:
            # text_embeds = self.embed_texts(data_dict['texts'])
            # print(input_ids)
            # print(text_embeds)
            # aa
            img_embeds = self.embed_images(img_preprocessed)
            img_embeds = self.image_projection(img_embeds)
            
            output = self.transformer(
                input_ids=input_ids.long(),
                key_value_states=img_embeds
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

        geojson_name = data_dict['geojson_name']

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
            if self.add_image_condition:
                # Use path relative to module location instead of current working directory
                debug_image_dir = Path(__file__).parent.parent / f'tiff_{geojson_name}_buf1'
                image_path = debug_image_dir / f"{data_dict['id']}.tif"
                bldg_image = Image.open(str(image_path))
                bldg_image = self.preprocessing(bldg_image).unsqueeze(0)
                # print(bldg_image, bldg_image.shape)

                net_device = next(self.parameters()).device
                bldg_image = bldg_image.to(net_device)

                image_embeds = self.embed_images(bldg_image)
                image_embeds = self.image_projection(image_embeds)
                condition = image_embeds
            # else:
            #     text_embeds = self.embed_texts(data_dict['texts'])
            #     condition = text_embeds
            # print(condition, condition.shape)

        # conditioned on 1/4 the shape
        input_ids = input_ids[:, attention_mask[0] == 1]  # 1 x [<bos> ... <eos>]
        num_faces = (input_ids.shape[1] - 2) // 9
        kept_length = (num_faces // 1) * 9 + 1
        input_ids = input_ids[:, :kept_length]  # 1 x [<bos> ...]

        net_device = next(self.parameters()).device
        max_length = 2306
        outputs = torch.ones(n_samples, max_length).long().to(net_device) * self.eos_token_id
        # batch x ntokens
        results = self.transformer.generate(
            input_ids=input_ids,
            key_value_states=condition, 
            max_new_tokens=max_length - input_ids.shape[1],
            do_sample=True,
            # top_k=10,
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
    
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
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
    
    # Clean up checkpoint from memory
    del checkpoint
    del available_weights
    torch.cuda.empty_cache()

    print(pretrained_ext)
    print('ckpt loaded. \n')
    
    return model.to(device)


def get_model(args):
    model = MeshXL(args)
    return model
