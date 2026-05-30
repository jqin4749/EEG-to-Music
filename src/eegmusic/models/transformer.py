import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

import torchvision
import math
from einops import rearrange
from dataclasses import dataclass
from ..utils.mask import random_masking
from ..utils.misc import init_pos_embedding1D
from torch.utils.checkpoint import checkpoint

@dataclass
class ModelOutput:
    last_hidden_state: torch.Tensor = None
    hidden_states: torch.Tensor = None
    attentions: torch.Tensor = None
    loss: torch.Tensor = None
    recon_signal: torch.Tensor = None
    feature_loss: torch.Tensor = None
    pixel_loss: torch.Tensor = None
    token_loss: torch.Tensor = None
    logits: torch.Tensor = None
    
class Attention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(emb_dim, emb_dim * 3, bias=False)
        self.fc_out = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_output = F.scaled_dot_product_attention(q, k, v)
        
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, emb_dim)
        out = self.fc_out(attn_output)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, emb_dim, ff_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(emb_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerLayer(nn.Module):

    def __init__(self, emb_dim, num_heads, ff_dim, dropout=0.):
        super(TransformerLayer, self).__init__()
        self.attn = Attention(emb_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.feed_forward = FeedForward(emb_dim, ff_dim, dropout)

    def forward(self, x):
        # Apply layer normalization before the attention and feed-forward layers
        attn_output = self.attn(self.norm1(x))
        x = x + attn_output
        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output
        return x

class Transformer(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, num_layers, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(emb_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.initialize_weights()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def _initialize_weights(self, module):
        # skip if not required_grad
        if not module.requires_grad_:
            return
        # Initialize weights for Linear layers using Xavier uniform distribution
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            # Initialize biases for Linear layers to zero
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Initialize weights for LayerNorm layers to one and biases to zero
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        # Initialize weights for Conv2d layers using Kaiming normal distribution
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            # Initialize biases for Conv2d layers to zero
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Initialize weights for Embedding layers with a normal distribution
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        # Initialize parameters with a normal distribution
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0, std=0.02)

    def initialize_weights(self):
        # Apply the weight initialization function to all modules
        self.apply(self._initialize_weights)

        
    
class LMAEConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super(LMAEConfig, self).__init__(**kwargs)
        self.encoder_emb_dim = kwargs.get("encoder_emb_dim", 1024)
        self.encoder_num_heads = kwargs.get("encoder_num_heads", 16)
        self.encoder_ff_dim = kwargs.get("encoder_ff_dim", 4096)
        self.encoder_num_layers = kwargs.get("encoder_num_layers", 12)

        self.decoder_emb_dim = kwargs.get("decoder_emb_dim", 1024)  
        self.decoder_num_heads = kwargs.get("decoder_num_heads", 16)
        self.decoder_ff_dim = kwargs.get("decoder_ff_dim", 4096)
        self.decoder_num_layers = kwargs.get("decoder_num_layers", 4)
        
        self.img_size = kwargs.get("img_size", 224)
        self.img_latent_layer = kwargs.get("img_latent_layer", "layer1") # layer1, layer2, layer3, layer4
        if 'img_latent_size' not in kwargs:
            if self.img_latent_layer == "layer1":
                self.img_latent_size = 56
                self.max_seq_len = 256
            elif self.img_latent_layer == "layer2":
                self.img_latent_size = 28
                self.max_seq_len = 512
            elif self.img_latent_layer == "layer3":
                self.img_latent_size = 14
                self.max_seq_len = 1024
            elif self.img_latent_layer == "layer4":
                self.img_latent_size = 7
                self.max_seq_len = 2048
            else:
                raise ValueError(f"Invalid image latent layer: {self.img_latent_layer}")
        else:
            self.img_latent_size = kwargs.get("img_latent_size", 56)
            self.max_seq_len = kwargs.get("max_seq_len", 256)

        self.dropout = kwargs.get("dropout", 0.1)



class LMAE_Encoder(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = LMAEConfig
    def __init__(self, config):
        super(LMAE_Encoder, self).__init__(config)
        self.max_seq_len = config.max_seq_len
        self.emb_dim = config.encoder_emb_dim
        self.pos_embedding = nn.Parameter(torch.zeros(config.max_seq_len, config.encoder_emb_dim)) 
        self.encoder = Transformer(config.encoder_emb_dim, config.encoder_num_heads, config.encoder_ff_dim, config.encoder_num_layers, config.dropout)
        init_pos_embedding1D(self.max_seq_len, config.encoder_emb_dim, self.pos_embedding)
    
    def forward(self, x, ids_keep=None):
        # x: [batch_size, seq_len, emb_dim]
        # ids_keep: [batch_size, seq_len]
        x = x + self.pos_embedding
        if ids_keep is not None:
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, self.emb_dim))
        x = self.encoder(x)
        return ModelOutput(last_hidden_state=x)

class GroupNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__()
        self.num_groups = self.get_num_groups(num_channels)
        self.group_norm = nn.GroupNorm(self.num_groups, num_channels, eps, affine)

    def get_num_groups(self, channels):
        """
        确保返回的groups数量能整除通道数
        Args:
            channels: 输入的通道数
        Returns:
            合适的groups数量
        """
        if channels <= 8:
            return 1
        
        # 从大到小尝试可能的groups数
        candidates = [32, 16, 8, 4, 2, 1]
        
        for num_groups in candidates:
            # 确保channels能被num_groups整除，且每组至少有8个通道
            if channels % num_groups == 0 and channels // num_groups >= 8:
                return num_groups
        
        # 如果上述都不满足，返回能整除channels的最大数
        # 获取所有可能的因子
        factors = [i for i in range(1, channels + 1) if channels % i == 0]
        # 返回不超过32的最大因子
        return max([f for f in factors if f <= 32])

    def forward(self, x):
        return self.group_norm(x)
    

class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super(ResnetBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = GroupNorm(num_channels=in_channels, eps=1e-5, affine=True)
        self.activation = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = GroupNorm(num_channels=out_channels, eps=1e-5, affine=True)
        self.dropout = nn.Dropout(dropout)
        
        # 添加维度调整层
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        
    def forward(self, x):
        # x: [batch_size, in_channels, height, width]
        identity = self.shortcut(x)
        
        hidden_states = self.norm1(x)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states += identity
        return hidden_states

class Upsample2D(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super(Upsample2D, self).__init__()
        self.channels = channels
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size=scale_factor, stride=scale_factor, bias=False)
        
    def forward(self, hidden_states):
        # hidden_states: [batch_size, channels, height, width]
        assert hidden_states.shape[1] == self.channels

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
        # hidden_states = F.interpolate(hidden_states, scale_factor=self.scale_factor, mode="bilinear")
        hidden_states = self.up(hidden_states)
        hidden_states = self.conv(hidden_states)
        return hidden_states

class UpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UpBlock2D, self).__init__()
        self.res1 = ResnetBlock2D(in_channels, out_channels, stride=1, dropout=dropout)
        self.res2 = ResnetBlock2D(out_channels, out_channels, stride=1, dropout=dropout)
        self.upsample = Upsample2D(out_channels, scale_factor=2)
        self.act = nn.SiLU()
        self.norm_out = GroupNorm(num_channels=out_channels, eps=1e-5)
    def forward(self, x):
        # x: [batch_size, in_channels, height, width]
        hidden_states = self.res1(x)
        hidden_states = self.res2(hidden_states)
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(self.norm_out(hidden_states))
        return hidden_states
        
class Feature2PixelModel(nn.Module):
    def __init__(self, in_channels, scale_factor=4, dropout=0.0):
        super(Feature2PixelModel, self).__init__()
        self.scale_factor = scale_factor
        n_up_blocks = math.ceil(math.log2(scale_factor))
        up_blocks = []
        for i in range(n_up_blocks):
            up_blocks.append(UpBlock2D(in_channels, in_channels // 2, dropout=dropout))
            in_channels = in_channels // 2
        self.up_blocks = nn.ModuleList(up_blocks)
        self.conv_out = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1) # 3 for RGB, kernel_size=3 & padding=1 for no change in resolution
        self.act = nn.SiLU()
        self.norm_out = GroupNorm(num_channels=in_channels, eps=1e-5)
    def forward(self, x):
        # x: [batch_size, in_channels, height, width]
        for up_block in self.up_blocks:
            x = up_block(x)
        # post-process
        x = self.act(self.norm_out(x))
        x = self.conv_out(x)
        return x
        
# feature inversion ?
class LMAE_Decoder(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = LMAEConfig
    def __init__(self, config):
        super(LMAE_Decoder, self).__init__(config)
        self.max_seq_len = config.max_seq_len
        self.emb_dim = config.decoder_emb_dim
        self.decoder = Transformer(config.decoder_emb_dim, config.decoder_num_heads, config.decoder_ff_dim, config.decoder_num_layers, config.dropout)
        self.pos_embedding = nn.Parameter(torch.zeros(config.max_seq_len, config.decoder_emb_dim)) 
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_emb_dim))
        init_pos_embedding1D(self.max_seq_len, config.decoder_emb_dim, self.pos_embedding)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.latent_embedding = nn.Linear(config.encoder_emb_dim, config.decoder_emb_dim)
        self.decoder_norm = nn.LayerNorm(config.decoder_emb_dim)
        self.token2feature = nn.Linear(config.decoder_emb_dim, config.img_latent_size**2)
        scale_factor = config.img_size // config.img_latent_size
   
        self.feature2pixel = Feature2PixelModel(config.max_seq_len, scale_factor, config.dropout)
        # self.token2pixel = nn.Linear(config.decoder_emb_dim,  3 * config.img_size * config.img_size)
        self.img_latent_size = config.img_latent_size
        self.img_size = config.img_size


    def forward(self, x, ids_restore=None):
        # x: [batch_size, seq_len, emb_dim]
        # embed tokens
        x = self.latent_embedding(x)

        # append mask tokens to sequence
        mask_token = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_token], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, self.emb_dim).to(x.device))
        x = x_ + self.pos_embedding

        x = self.decoder(x)
        x = self.decoder_norm(x)
        # x = rearrange(x, 'b q b -> b (q b)')
        # x = x.mean(dim=1)
        # x = self.token2pixel(x) # [batch_size, 3 * img_size * img_size]
        # x = rearrange(x, 'b (c h w) -> b c h w', c=3, h=self.img_size, w=self.img_size)
        # import pdb; pdb.set_trace()
        x = self.token2feature(x)
        features = rearrange(x, 'b c (h w) -> b c h w', h=self.img_latent_size, w=self.img_latent_size)
        x = self.feature2pixel(features)
        return x
        
        
class LMAEForMaskedTraining(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = LMAEConfig
    def __init__(self, config):
        super(LMAEForMaskedTraining, self).__init__(config)
        self.encoder = Transformer(config.encoder_emb_dim, config.encoder_num_heads, config.encoder_ff_dim, config.encoder_num_layers, config.dropout)
        self.latent_embedding = nn.Linear(config.img_latent_size**2, config.encoder_emb_dim)
        self.max_seq_len = config.max_seq_len
        self.pos_embedding = nn.Parameter(torch.zeros(config.max_seq_len, config.encoder_emb_dim)) 
        self.img_latent_layer = config.img_latent_layer
        self.resent = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        # self.resent.eval()
        self.encoder_norm = nn.LayerNorm(config.encoder_emb_dim)
        self.resnet_norm = nn.LayerNorm(config.img_latent_size**2)

        init_pos_embedding1D(self.max_seq_len, config.encoder_emb_dim, self.pos_embedding)
        # self.resent.requires_grad = False
        # self.latent_embedding.requires_grad = False # # TODO: use random token projection ?
        self.encoder_emb_dim = config.encoder_emb_dim

        # decoder
        self.decoder = LMAE_Decoder(config)

        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        # Register hooks for different layers
        self.resent.layer1[-1].register_forward_hook(get_activation('layer1'))  # Early features
        self.resent.layer2[-1].register_forward_hook(get_activation('layer2'))  # Mid-level features 
        self.resent.layer3[-1].register_forward_hook(get_activation('layer3'))  # Higher-level features
        self.resent.layer4[-1].register_forward_hook(get_activation('layer4'))  # Final conv features

    
    def forward_resnet(self, x):
        # x: [batch_size, 3, 224, 224]
        _ = self.resent(x)
        if self.img_latent_layer == "layer1":
            act = self.activation['layer1'] # [batch_size, 256, 56, 56]
        elif self.img_latent_layer == "layer2":
            act = self.activation['layer2'] # [batch_size, 512, 28, 28]
        elif self.img_latent_layer == "layer3":
            act = self.activation['layer3'] # [batch_size, 1024, 14, 14]
        elif self.img_latent_layer == "layer4":
            act = self.activation['layer4'] # [batch_size, 2048, 7, 7]
        else:
            raise ValueError(f"Invalid image latent layer: {self.img_latent_layer}")
        
        act = rearrange(act, 'b c h w -> b c (h w)')
        self.activation = {}
        return act # [batch_size, seq_len, emb_dim]
    
    def forward_encoder(self, x): 
        # No masking
        hidden_states = self.forward_resnet(x)
        hidden_states = self.latent_embedding(hidden_states)
        hidden_states = hidden_states + self.pos_embedding
        hidden_states = self.encoder(hidden_states)     
        return ModelOutput(last_hidden_state=hidden_states)
    
    def forward(self, x, mask_ratio=0.75):
        # x: [batch_size, 3, 224, 224]
        resnet_features = self.forward_resnet(x)
        resnet_features = self.resnet_norm(resnet_features)
    
        hidden_states = self.latent_embedding(resnet_features)
        hidden_states = hidden_states + self.pos_embedding
        _, ids_restore, ids_keep = random_masking(self.max_seq_len, x.shape[0], mask_ratio)
        hidden_states = torch.gather(hidden_states, dim=1, 
                                     index=ids_keep.unsqueeze(-1).repeat(1, 1, self.encoder_emb_dim).to(hidden_states.device))
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.encoder_norm(hidden_states)
        # import pdb; pdb.set_trace()
        recon_x = self.decoder(hidden_states, ids_restore) # [batch_size, 3, img_size, img_size]
        loss = F.mse_loss(recon_x, x)

        return ModelOutput(loss=loss, recon_signal=recon_x)

