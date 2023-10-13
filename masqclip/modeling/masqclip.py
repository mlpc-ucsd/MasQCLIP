from collections import OrderedDict
from typing import List, Tuple, Union

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import clip
from torchvision.transforms.transforms import Normalize

# datasets
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from ..data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
from ..data.datasets.register_ade20k_full import ADE20K_SEM_SEG_FULL_CATEGORIES
from ..data.datasets.register_pascal_semseg import PASCAL_59_CATEGORIES
from ..data.datasets.register_pascal_full_semseg import PASCAL_459_CATEGORIES
from ..data.datasets.register_coco_split import COCO_CATEGORIES_INSTANCES65, COCO_CATEGORIES_BASE48


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, d_model: int, n_head: int):
        super().__init__(d_model, n_head)
        assert self._qkv_same_embed_dim
        self.new_q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, query, nq, attn_mask, need_weights=False):
        seq, bs, _ = query.shape
    
        # [Mask Class Tokens, (class token, image tokens)]
        q, k, v = F.linear(query[nq:].detach(), self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1)
        q = q / math.sqrt(self.head_dim)
        
        clip_attn = torch.bmm(q, k.transpose(-2, -1))
        clip_output = torch.bmm(F.softmax(clip_attn, dim=-1), v)
        clip_output = clip_output.transpose(0, 1).reshape(-1, bs, self.embed_dim)
        
        assert attn_mask.dtype == torch.bool
        attn_mask_float = torch.zeros_like(attn_mask, dtype=q.dtype)
        attn_mask_float = attn_mask_float.masked_fill(attn_mask, float("-inf"))
        
        # Mask Class Tokens
        new_q = self.new_q_proj(query[:nq])
        new_q = new_q.contiguous().view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1)
        new_q = new_q / math.sqrt(self.head_dim)

        mask_attn = torch.bmm(new_q, k.transpose(-2, -1))
        mask_output = torch.bmm(F.softmax(mask_attn + attn_mask_float, dim=-1), v)
        mask_output = mask_output.transpose(0, 1).reshape(nq, bs, self.embed_dim)

        attn_output = torch.concat([mask_output, clip_output], dim=0).contiguous()
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(seq, bs, -1)

        if need_weights:
            attn_output_weights = mask_attn.view(bs, self.num_heads, nq, -1)
            attn_output_weights = attn_output_weights.mean(dim=1)
            return attn_output, attn_output_weights
        else:
            return attn_output, None


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, y, nq, attn_mask):
        return self.attn(y, nq, attn_mask, need_weights=False)[0]

    def forward(self, y, attn_mask):
        bs, nq, _ = attn_mask.shape
        attn_mask = attn_mask[:, None].repeat(1, self.n_head, 1, 1)
        attn_mask = attn_mask.view(bs * self.n_head, nq, -1)

        y = y + self.attention(self.ln_1(y), nq, attn_mask)
        y = y + self.mlp(self.ln_2(y))
        return y


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, patch_size: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.patch_size = patch_size
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])
    
    def forward(self, y, attn_mask):
        for layer in list(self.resblocks.modules())[0]:
            y = layer(y, attn_mask)
        return y


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.layers = layers
        self.width = width

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, patch_size)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # normalize
        self.clip_prep_img = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def forward(self, img, masks, mask_pe):
        img_size = self.input_resolution
        x = F.interpolate(img / 255., (img_size, img_size), mode="bicubic")
        x = self.clip_prep_img(x)
        
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        clip_token = x + self.positional_embedding.to(x.dtype)
        mask_token = mask_pe + self.class_embedding  # Mask Class Tokens
        tokens = torch.concat([mask_token, clip_token], dim=1)
        
        attn_mask = self.get_attn_masks(masks)
        tokens = self.ln_pre(tokens).permute(1, 0, 2)  # NLD -> LND
        tokens = self.transformer(tokens, attn_mask)
        return tokens

    def get_final_embedding(self, tokens, nq: int):
        tokens = tokens.permute(1, 0, 2)  # LND -> NLD
        embedding = self.ln_post(tokens[:, :nq])
        embedding = embedding @ self.proj
        return embedding

    def get_attn_masks(self, pred_masks):
        img_size = self.input_resolution
        masks = F.interpolate(pred_masks, (img_size, img_size), mode="bilinear")
        masks = F.max_pool2d(masks, self.patch_size, self.patch_size)
        bin_masks = (masks > 0.).flatten(2)  # binary
        attn_mask = torch.concat((torch.ones_like(bin_masks[..., [0]]), bin_masks), dim=2)
        return attn_mask.logical_not()


class MasQCLIP(nn.Module):
    def __init__(self, dataset_name, model_name):
        super().__init__()
        assert len(model_name) == 1
        _clip, _ = clip.load(model_name[0], device="cuda")
        self.visual = self.load_clip_model(model_name[0])
        keys = self.visual.load_state_dict(_clip.visual.state_dict(), strict=False)
        
        # text embeddings
        with torch.no_grad():
            self.text_embeddings = self.load_text_embedding(dataset_name, _clip)
            
        # positional embedding
        self.mask_embeddings = nn.Parameter(self.visual.positional_embedding[0])
        del _clip, keys

    def forward(self, img, masks):
        bs, nq, device = masks.shape[0], masks.shape[1], img.device
        mask_pe = self.mask_embeddings.to(device) + torch.zeros((bs, nq, self.visual.width), device=device)
        tokens = self.visual(img, masks, mask_pe)

        # projection
        feature = self.visual.get_final_embedding(tokens, nq)
        feature = feature / feature.norm(p=2, dim=-1, keepdim=True)
        pred_logits = torch.einsum("bqc,nc->bqn", feature, self.text_embeddings) * 100
        return {"pred_logits": pred_logits}

    def load_clip_model(self, model_name):
        if model_name == "ViT-L/14@336px":
            return VisionTransformer(input_resolution=336, patch_size=14, width=1024, layers=24, heads=16, output_dim=768)
        elif model_name == "ViT-L/14":
            return VisionTransformer(input_resolution=224, patch_size=14, width=1024, layers=24, heads=16, output_dim=768)
        elif model_name == "ViT-B/16":
            return VisionTransformer(input_resolution=224, patch_size=16, width=768, layers=12, heads=12, output_dim=512)
        elif model_name == "ViT-B/32":
            return VisionTransformer(input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512)
        
        assert False

    def load_text_embedding(self, dataset_name, model):
        if "coco_2017" in dataset_name:  # COCO-panoptic training
            class_names = [x["name"] for x in COCO_CATEGORIES] + ["background"]
            return self.load_coco_text_embedding(class_names, model, "cuda").float().detach()
        elif "ade20k_full" in dataset_name:  # ADE20K-847 testing
            class_names = [x["name"] for x in ADE20K_SEM_SEG_FULL_CATEGORIES] + ["background"]
            return self.load_ade20k_text_embedding(class_names, model, "cuda").float().detach()
        elif "ade20k" in dataset_name:  # ADE20K-150 testing
            class_names = [x["name"] for x in ADE20K_150_CATEGORIES] + ["background"]
            return self.load_ade20k_text_embedding(class_names, model, "cuda").float().detach()
        elif "pascal_context_59" in dataset_name:  # pascal-context-59 testing
            class_names = [x["name"] for x in PASCAL_59_CATEGORIES] + ["background"]
            return self.load_pascal_context_text_embedding(class_names, model, "cuda").float().detach()
        elif "pascal_context_459" in dataset_name:  # pascal-context-459 testing
            class_names = PASCAL_459_CATEGORIES + ["background"]
            return self.load_pascal_context_text_embedding(class_names, model, "cuda").float().detach()
        elif "instances65" in dataset_name:  # COCO-instance
            class_names = [x["name"] for x in COCO_CATEGORIES_INSTANCES65] + ["background"]
            return self.load_coco_ins_text_embedding(class_names, model, "cuda").float().detach()
        elif "base48" in dataset_name:  # COCO-base48
            class_names = [x['name'] for x in COCO_CATEGORIES_BASE48] + ["background"]
            return self.load_coco_ins_text_embedding(class_names, model, "cuda").float().detach()
        
        assert False

    def load_coco_text_embedding(self, classes, model, device):
        class_map = {
            "door-stuff": ["door"],
            "floor-wood": ["wood floor"],
            "mirror-stuff": ["mirror"],
            "wall-brick": ["brick wall"],
            "wall-stone": ["stone wall"],
            "wall-tile": ["wall tile"],
            "wall-wood": ["wood wall"],
            "water-other": ["water"],
            "window-blind": ["window blind"],
            "window-other": ["window"],
            "tree-merged": ["branch", "tree", "bush", "leaves"],
            "fence-merged": ["cage", "fence", "railing"],
            "ceiling-merged": ["ceiling tile", "ceiling"],
            "sky-other-merged": ["clouds", "sky", "fog"],
            "cabinet-merged": ["cupboard", "cabinet"],
            "table-merged": ["desk stuff", "table"],
            "floor-other-merged": ["marble floor", "floor", "floor tile"],
            "pavement-merged": ["stone floor", "pavement"],
            "mountain-merged": ["hill", "mountain"],
            "grass-merged": ["moss", "grass", "straw"],
            "dirt-merged": ["mud", "dirt"],
            "paper-merged": ["napkin", "paper"],
            "food-other-merged": ["salad", "vegetable", "food"],
            "building-other-merged": ["skyscraper", "building"],
            "rock-merged": ["stone", "rock"],
            "wall-other-merged": ["wall", "concrete wall", "panel wall"],
            "rug-merged": ["mat", "rug", "carpet"],
        }

        def get_class_embedding(classes, model, device):  # coco
            class_embedding = []
            for class_name in classes:
                if class_name in class_map.keys():
                    class_name = class_map[class_name]
                text_embedding = model.encode_text(clip.tokenize(class_name).to(device))
                text_embedding = text_embedding.mean(dim=0, keepdim=True)
                class_embedding.append(text_embedding)
            class_embedding = torch.concat(class_embedding, dim=0).float().detach()
            class_embedding = class_embedding / class_embedding.norm(p=2, dim=-1, keepdim=True)
            return class_embedding
        
        return get_class_embedding(classes, model, device)
        
    def load_ade20k_text_embedding(self, classes, model, device):
        class_embedding = []
        class_names = [c.split(", ") for c in classes]
        for names in class_names:
            text_embedding = model.encode_text(clip.tokenize(names).to(device))
            text_embedding = text_embedding.mean(dim=0, keepdim=True)
            class_embedding.append(text_embedding)
        class_embedding = torch.concat(class_embedding, dim=0).float().detach()
        class_embedding = class_embedding / class_embedding.norm(p=2, dim=-1, keepdim=True)
        return class_embedding
    
    def load_pascal_context_text_embedding(self, classes, model, device):
        tokens = clip.tokenize(classes).to(device)
        class_embedding = model.encode_text(tokens).float().detach()
        class_embedding = class_embedding / class_embedding.norm(p=2, dim=-1, keepdim=True)
        return class_embedding
    
    def load_coco_ins_text_embedding(self, classes, model, device):
        tokens = clip.tokenize(classes).to(device)
        class_embedding = model.encode_text(tokens).float().detach()
        class_embedding = class_embedding / class_embedding.norm(p=2, dim=-1, keepdim=True)
        return class_embedding
