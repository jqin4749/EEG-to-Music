import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .transformer import Transformer
from dataclasses import dataclass
from transformers import PreTrainedModel, PretrainedConfig
from ..utils.mask import random_masking
from ..utils.misc import calculate_parameters
import numpy as np
from .common import ModelOutput
from .stat_model import StatHead
from .whisper import WhisperEncoder, CLAPEncoder
from tqdm import tqdm
from sklearn.linear_model import Ridge
from audioldm import build_model
import os
import json
import torch.fft

class EEGEncoderConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super(EEGEncoderConfig, self).__init__(**kwargs)
        self.embed_dim = kwargs.get("embed_dim", 768)
        self.patch_size = kwargs.get("patch_size", 100)
        self.eeg_length = kwargs.get("eeg_length", 1600)
        self.num_channels = kwargs.get("num_channels", 125)
        self.channel_dropout = kwargs.get("channel_dropout", 0.0)
        self.aligned_space_dim = kwargs.get("aligned_space_dim", 1024)
        self.num_heads = kwargs.get("num_heads", 16)
        self.num_layers = kwargs.get("num_layers", 12)

class ChannelDropout(nn.Module):
    def __init__(self, p=0.0):
        super(ChannelDropout, self).__init__()
        self.p = p
    
    def forward(self, x, p=None):
        # x: [b, c, h, w]
        p = self.p if p is None else p
        if self.training and p > 0:
            b, c, h, w = x.shape
            mask, ids_restore, ids_keep = random_masking(c, b, p)
            x = torch.gather(x, dim=1, index=ids_keep[:, :, None, None].repeat(1, 1, h, w).to(x.device))
        return x

class EEGEncoderForPretrain(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = EEGEncoderConfig

    def __init__(self, config):
        super(EEGEncoderForPretrain, self).__init__(config)
        self.patch_embed = nn.Conv2d(1, config.embed_dim, kernel_size=(1, config.patch_size), stride=(1, config.patch_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_channels, config.eeg_length // config.patch_size, config.embed_dim), 
                                      requires_grad=True) # 125 channels, 1600 time points
        self.transformer = Transformer(emb_dim=config.embed_dim, num_heads=config.num_heads, 
                                       ff_dim=config.embed_dim*4, num_layers=config.num_layers, dropout=0.)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim), requires_grad=True)
        self.out_norm = nn.LayerNorm(config.embed_dim)
        self.channel_dropout = ChannelDropout(p=config.channel_dropout)
        # self.algined_space_proj = nn.Linear(config.embed_dim, config.aligned_space_dim)
        self.aligned_space_proj = DINOHead(config.embed_dim, out_dim=config.aligned_space_dim)

        # initialize pos_embed
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        # nn.init.trunc_normal_(self.algined_space_proj.weight, std=.02)
        # self.algined_space_proj.bias.data.zero_()

        num_parameters = calculate_parameters(self)
        print(f"Number of trainable parameters: {num_parameters/1e6:.2f}M")

    def forward_encoder(self, x, channel_dropout=0.0):
        x = x.unsqueeze(1) # [batch_size, 1, 125, ts_len]
        x = self.patch_embed(x) # [batch_size, embed_dim, 125, ts_len // patch_size]
        x = rearrange(x, 'b c h w -> b h w c') # [batch_size, 125, ts_len // patch_size, embed_dim]
        x = x + self.pos_embed[:, :, :x.shape[2], :] # allow for variable length EEG
        x = self.channel_dropout(x, p=channel_dropout)
        x = rearrange(x, 'b h w c -> b (h w) c') 
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) 
        x = self.transformer(x)
        x = self.out_norm(x)
        return x
    
    def forward_multiview(self, x, channel_dropout=0.0):
        # x is a list of tensors of shape [batch_size, 125, ts_len*]
        # return latent feature of dimension [num_views, batch_size, embed_dim]
        x = [self.forward_encoder(x_i, channel_dropout)[:, 0, :] for x_i in x]
        x = torch.stack(x)
        aligned_space = self.aligned_space_proj(x) # [num_views, batch_size, aligned_space_dim]
        return ModelOutput(last_hidden_state=x,
                           aligned_space=aligned_space)

    def forward(self, x, channel_dropout=0.0):
        if isinstance(x, list):
            return self.forward_multiview(x, channel_dropout)
        # x: [batch_size, 125, ts_len]
        x = self.forward_encoder(x, channel_dropout)
        aligned_space = self.aligned_space_proj(x[:, 0, :])

        return ModelOutput(last_hidden_state=x[:, 0, :], 
                           hidden_states=x[:, 1:, :], 
                           aligned_space=aligned_space
                           )

class EEGEncoderForAlign(EEGEncoderForPretrain):
    def __init__(self, config):
        super(EEGEncoderForAlign, self).__init__(config)
        self.mlp = nn.Linear(config.embed_dim, config.embed_dim)
        self.aligned_space_proj = nn.Linear(config.embed_dim, config.aligned_space_dim)
       
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.out_norm_aligned_space = nn.LayerNorm(config.aligned_space_dim)

        self.whisper = WhisperEncoder(model_name='openai/whisper-base', aligned_space_dim=config.aligned_space_dim)
        # self.whisper = CLAPEncoder(aligned_space_dim=config.aligned_space_dim)
        # self.whisper.model.requires_grad = False
   
        nn.init.trunc_normal_(self.aligned_space_proj.weight, std=.02)
        self.aligned_space_proj.bias.data.zero_()
        nn.init.trunc_normal_(self.mlp.weight, std=.02)
        self.mlp.bias.data.zero_()

        num_parameters = calculate_parameters(self)
        print(f"Number of trainable parameters: {num_parameters/1e6:.2f}M")

    @classmethod
    def from_pretrained(self, path):
        model = super().from_pretrained(path)
        model.whisper = WhisperEncoder(model_name='openai/whisper-base', aligned_space_dim=model.config.aligned_space_dim)
        # model.whisper = CLAPEncoder(aligned_space_dim=model.config.aligned_space_dim)
        model.whisper.model.requires_grad = False
        return model

    @classmethod
    def from_pretrained_with_whisper(self, path):
        model = super().from_pretrained(path)
        return model
    
    def forward(self, eeg, music, music_prepocessed=None, channel_dropout=0.0):
        eeg_features = self.forward_eeg_encoder(eeg, channel_dropout)
        if music_prepocessed is None:
            audio_features = self.forward_whisper(music)
        else:
            audio_features = self.whisper(music_prepocessed)
      
        loss, acc = self.loss_fn(eeg_features, audio_features, return_acc=True)
        # loss, acc = self.loss_fn_cov(eeg_features, audio_features, return_acc=True)

        return ModelOutput(
            loss = loss,
            acc = acc
        )
    
    # @torch.no_grad()
    def forward_encoder(self, x, channel_dropout=0.0):
        return super().forward_encoder(x, channel_dropout)
    
    def forward_eeg_encoder(self, x, channel_dropout=0.0):
        x = self.forward_encoder(x, channel_dropout)
        aligned_space = F.gelu(self.mlp(x.mean(dim=-2)))
        aligned_space = self.aligned_space_proj(aligned_space)
        aligned_space = self.out_norm_aligned_space(aligned_space)
        return aligned_space
    
    def forward_whisper(self, x):
        x = self.whisper.preprocess(x).to(self.device)
        x = self.whisper(x)
        return x

    def loss_fn_cov(self, eeg_features, audio_features, return_acc=False):
        logit_scale = 1 # self.logit_scale.exp()
        eeg_features = F.normalize(logit_scale * eeg_features, p=2, dim=-1)
        audio_features = F.normalize(logit_scale * audio_features, p=2, dim=-1)
        cov = eeg_features @ audio_features.T
        identity = torch.eye(cov.shape[0]).to(self.device) + 1e-6
        loss = torch.mean((cov - identity) ** 2)
        if return_acc:
            acc = (cov.argmax(dim=-1) == torch.arange(cov.shape[0]).to(self.device)).float().sum() / cov.shape[0]
            return loss, acc
        return loss
    
    def loss_fn(self, eeg_features, audio_features, return_acc=False):
        # clip contrastive loss
        # import pdb; pdb.set_trace()
        logit_scale = self.logit_scale.exp()
        eeg_features = F.normalize(eeg_features, p=2, dim=-1)
        audio_features = F.normalize(audio_features, p=2, dim=-1)
        logits_per_eeg = logit_scale * eeg_features @ audio_features.T
        logits_per_audio = logits_per_eeg.T
        loss = F.cross_entropy(logits_per_eeg, torch.arange(logits_per_eeg.shape[0]).to(self.device)) 
        loss += F.cross_entropy(logits_per_audio, torch.arange(logits_per_audio.shape[0]).to(self.device))
        if loss.isnan():
            loss = 0.0
            print("Loss is nan, setting to 0.0")
            print(f"Eeg features: {eeg_features.shape}, Audio features: {audio_features.shape}")
            print(eeg_features, audio_features)
            exit(0)

        if return_acc:
            acc = (logits_per_eeg.argmax(dim=-1) == torch.arange(logits_per_eeg.shape[0]).to(self.device)).float().sum() / logits_per_eeg.shape[0]
            return loss / 2.0, acc
        return loss / 2.0

class LinearFourier(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super(LinearFourier, self).__init__()
        self.mlp_real = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        self.mlp_imag = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        # nn.init.trunc_normal_(self.mlp_imag.weight, std=.02)
        # self.mlp_real.bias.data.zero_()
        # self.mlp_imag.bias.data.zero_()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # x: torch.cfloat
        real = self.mlp_real(x.real)
        imag = self.mlp_imag(x.imag)
        real = self.dropout(real)
        imag = self.dropout(imag)
        x = real + imag * 1j
        return x

# class LinearFourier(nn.Module):
#     def __init__(self, in_dim, out_dim, dropout=0.0):
#         super(LinearFourier, self).__init__()
#         self.mlp = nn.Linear(in_dim * 2, in_dim * 2)
#         self.mlp1 = nn.Linear(in_dim * 2, in_dim * 2)
#         self.mlp2 = nn.Linear(in_dim * 2, out_dim)
#         # initialize with random complex weights
#         nn.init.trunc_normal_(self.mlp.weight, std=.02)
#         self.mlp.bias.data.zero_()
#         nn.init.trunc_normal_(self.mlp1.weight, std=.02)
#         self.mlp1.bias.data.zero_()
#         nn.init.trunc_normal_(self.mlp2.weight, std=.02)
#         self.mlp2.bias.data.zero_()
#         self.dropout = nn.Dropout(dropout)
    
#     def l1_regularization_loss(self, l1_regularization):
#         return l1_regularization * (torch.sum(torch.abs(self.mlp.weight)) 
#                                          + torch.sum(torch.abs(self.mlp1.weight)) 
#                                          + torch.sum(torch.abs(self.mlp2.weight)))

#     def forward(self, x):
#         # x: torch.cfloat
#         real_imag = torch.cat([x.real, x.imag], dim=-1)
#         real_imag = self.mlp(real_imag)
#         real_imag = F.gelu(real_imag)
#         real_imag = self.dropout(real_imag)
#         real_imag = self.mlp1(real_imag)
#         real_imag = F.gelu(real_imag)
#         real_imag = self.dropout(real_imag)
#         real_imag = self.mlp2(real_imag)
#         return real_imag
    
class EEGEncoderForAlignFourier(EEGEncoderForAlign):
    def __init__(self, config):
        super(EEGEncoderForAlignFourier, self).__init__(config)
        self.aligned_space_proj = LinearFourier(config.embed_dim, config.aligned_space_dim)

        self.whisper = WhisperEncoderFourier(model_name='openai/whisper-base', aligned_space_dim=config.aligned_space_dim)
        self.whisper.model.requires_grad = False

    @torch.no_grad()
    def forward_encoder(self, x, channel_dropout=0.0):
        x = x.unsqueeze(1) # [batch_size, 1, 125, ts_len]
        x = self.patch_embed(x) # [batch_size, embed_dim, 125, ts_len // patch_size]
        x = rearrange(x, 'b c h w -> b h w c') # [batch_size, 125, ts_len // patch_size, embed_dim]
        num_patches = x.shape[2]
        x = x + self.pos_embed[:, :, :num_patches, :] # allow for variable length EEG
        x = self.channel_dropout(x, p=channel_dropout)
        x = rearrange(x, 'b h w c -> b (h w) c') 
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) 
        x = self.transformer(x)
        x = self.out_norm(x)
        # x = rearrange(x[:, 1:, :], 'b (h w) c -> b h w c', w=num_patches)
        return x
    
    def forward_eeg_encoder(self, x, channel_dropout=0.0):
        x = self.forward_encoder(x, channel_dropout)
        x = torch.fft.fft(x, dim=1)
        # aligned_space = rearrange(aligned_space, 'b h w c -> b (h w) c')
        aligned_space = x.mean(dim=-2)
        aligned_space = self.aligned_space_proj(aligned_space)
        return aligned_space
    
    @classmethod
    def from_pretrained(self, path):
        model = super().from_pretrained(path)
        model.whisper = WhisperEncoderFourier(model_name='openai/whisper-base', aligned_space_dim=model.config.aligned_space_dim)
        # model.whisper = CLAPEncoder(aligned_space_dim=model.config.aligned_space_dim)
        model.whisper.model.requires_grad = False
        return model
    
    @classmethod
    def from_pretrained_with_whisper(self, path):
        model = super().from_pretrained_with_whisper(path)
        return model
    
    def complex_matmul(self, x, y):
        # x: [b, w]
        # y: [b, w]
        # return: [b, b]
        # transpose complex conjugate of y
        y_t = y.transpose(-2, -1).conj()
        return x @ y_t
    
    def complex_norm(self, x):
        # x: [b, w]
        # return: [b, w]
        x_conj = x.conj()
        norm = torch.sqrt(torch.sum(x * x_conj, dim=-1, keepdim=True))
        return x / norm


    def loss_fn_cov(self, eeg_features, audio_features, return_acc=False):
        # clip contrastive loss
        logit_scale = self.logit_scale.exp()
        if eeg_features.dtype == torch.cfloat:
            eeg_features = self.complex_norm(eeg_features)
            audio_features = self.complex_norm(audio_features)
            logits_per_eeg = logit_scale * self.complex_matmul(eeg_features, audio_features).abs()
            logits_per_audio = logits_per_eeg.T
        else:
            eeg_features = F.normalize(eeg_features, p=2, dim=-1)
            audio_features = F.normalize(audio_features, p=2, dim=-1)
            logits_per_eeg = logit_scale * eeg_features @ audio_features.T
            logits_per_audio = logits_per_eeg.T
        identity = torch.eye(logits_per_eeg.shape[0]).to(self.device) + 1e-6
        loss = torch.mean((logits_per_eeg - identity) ** 2)
        # loss = F.cross_entropy(logits_per_eeg, torch.arange(logits_per_eeg.shape[0]).to(self.device)) 
        # loss += F.cross_entropy(logits_per_audio, torch.arange(logits_per_audio.shape[0]).to(self.device))
        # loss = loss / 2.0
        # loss += self.aligned_space_proj.l1_regularization_loss(0.00002)
        if return_acc:
            acc = (logits_per_eeg.argmax(dim=-1) == torch.arange(logits_per_eeg.shape[0]).to(self.device)).float().sum() / logits_per_eeg.shape[0]
            return loss, acc
        return loss
    
    def save_pretrained(self, path):
        torch.save(self.state_dict(), os.path.join(path, 'model.pt'))
        self.config.save_pretrained(path)

    # @classmethod
    # def from_pretrained(cls, path):
    #     config = EEGEncoderConfig.from_pretrained(path)
    #     model = cls(config)
    #     missing_keys, unexpected_keys = model.load_state_dict(torch.load(os.path.join(path, 'model.pt')), strict=False)
    #     print(f"Missing keys: {missing_keys}")
    #     print(f"Unexpected keys: {unexpected_keys}")
    #     return model
    
class EEGEncoderForAlignLn(EEGEncoderForAlign):
    def __init__(self, config):
        super(EEGEncoderForAlignLn, self).__init__(config)
        # self.aligned_space_proj = DINOHead(config.embed_dim, out_dim=config.aligned_space_dim)
        self.aligned_space_proj = StatHead(config.embed_dim, config.aligned_space_dim)

    # @torch.no_grad()
    def forward_encoder(self, x, channel_dropout=0.0):
        return super().forward_encoder(x, channel_dropout)
    
    def forward_eeg_encoder(self, x, channel_dropout=0.0):
        x = self.forward_encoder(x, channel_dropout) # [batch_size, seq_len*, embed_dim]
        aligned_space = self.aligned_space_proj(x)
        aligned_space = self.out_norm_aligned_space(aligned_space)
        return aligned_space

class EEGEncoder(EEGEncoderForPretrain):
    def __init__(self, config):
        super(EEGEncoder, self).__init__(config)
        self.aligned_space_proj = nn.Linear(config.embed_dim, config.aligned_space_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.out_norm_aligned_space = nn.LayerNorm(config.aligned_space_dim)   

    def forward(self, eeg, channel_dropout=0.0):
        x = self.forward_encoder(eeg, channel_dropout)
        aligned_space = self.aligned_space_proj(x[:, 0, :])
        aligned_space = self.out_norm_aligned_space(aligned_space)
   
        return ModelOutput(
            last_hidden_state = aligned_space,
            hidden_states=x[:, 1:, :]
        )
    
class EEGEncoderLn(EEGEncoder):
    def __init__(self, config):
        super(EEGEncoderLn, self).__init__(config)
        self.aligned_space_proj = StatHead(config.embed_dim, config.aligned_space_dim) 

    def forward(self, eeg, channel_dropout=0.0):
        x = self.forward_encoder(eeg, channel_dropout)
        aligned_space = self.aligned_space_proj(x)
        aligned_space = self.out_norm_aligned_space(aligned_space)
   
        return ModelOutput(
            last_hidden_state = aligned_space,
            hidden_states=x[:, 1:, :]
        )
    
class EEGEncoderLnFourier(EEGEncoder):
    def __init__(self, config):
        super(EEGEncoderLnFourier, self).__init__(config)
        self.aligned_space_proj = nn.Sequential(
            LinearFourier(config.embed_dim, config.embed_dim, activation=nn.GELU(), dropout=0.0),
            LinearFourier(config.embed_dim, config.embed_dim, activation=nn.GELU(), dropout=0.0),
            LinearFourier(config.embed_dim, config.aligned_space_dim)
        )

    def forward(self, eeg, channel_dropout=0.0):
        x = self.forward_encoder(eeg, channel_dropout)
        x = torch.fft.fft(x, dim=1)
        x = x.mean(dim=-2)
        aligned_space = self.aligned_space_proj(x)
        return ModelOutput(
            last_hidden_state = aligned_space
        )
    @classmethod
    def from_pretrained(cls, path):
        config = EEGEncoderConfig.from_pretrained(path)
        model = cls(config)
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(os.path.join(path, 'model.pt')), strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        return model
    
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class DINOLoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        # teacher_out = (teacher_output - self.center) 
        teacher_out = teacher_out.detach()

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                # if v == iq:
                #     # we skip cases where student and teacher operate on the same view
                #     continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                # loss = 1 - F.cosine_similarity(student_out[v], q, dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        teacher_output = rearrange(teacher_output, 'v b c -> (v b) c')
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)



class EEGRidge:
    def __init__(self, eeg_dataloader, music_features, device='cuda'):
        self.eeg_dataloader = eeg_dataloader
        self.music_features = music_features
        self.clf = Ridge(alpha=0.1)
        eeg_features = []
        for batch in tqdm(self.eeg_dataloader):
            eeg_features.append(batch['eeg'].detach().cpu().numpy())
        eeg_features = np.concatenate(eeg_features, axis=0)
        self.clf.fit(eeg_features.reshape(eeg_features.shape[0], -1), self.music_features)
        pipeline = build_model(model_name="audioldm-m-full")
        audio_model = pipeline.cond_stage_model
        audio_model = audio_model.eval().to(device)
        audio_model.embed_mode = 'audio'
        audio_model.unconditional_prob = 0.0
        self.audio_model = audio_model
        self.device = device
    def predict(self, eeg_features):
        eeg_features = eeg_features.reshape(eeg_features.shape[0], -1)
        return self.clf.predict(eeg_features)

    @torch.no_grad()
    def get_music_features(self, music):
        music = (music - music.mean(dim=1, keepdim=True)) / (torch.max(torch.abs(music), dim=1, keepdim=True).values + 1e-6)
        music = music * 0.5
        audio_features = self.audio_model(music.to(self.device))[:,0]
        return audio_features
    
    def loss_fn(self, eeg_features, audio_features, return_acc=False, logit_scale=3.0):
        # clip contrastive loss
        eeg_features = F.normalize(eeg_features, p=2, dim=-1)
        audio_features = F.normalize(audio_features, p=2, dim=-1)
        logits_per_eeg = logit_scale * eeg_features @ audio_features.T
        logits_per_audio = logits_per_eeg.T
        loss = F.cross_entropy(logits_per_eeg, torch.arange(logits_per_eeg.shape[0]).to(self.device)) 
        loss += F.cross_entropy(logits_per_audio, torch.arange(logits_per_audio.shape[0]).to(self.device))
        if return_acc:
            acc = (logits_per_eeg.argmax(dim=-1) == torch.arange(logits_per_eeg.shape[0]).to(self.device)).float().sum() / logits_per_eeg.shape[0]
            return loss / 2.0, acc
        return loss / 2.0
    
    def __call__(self, eeg, music, channel_dropout=0.0):
        music_features = self.get_music_features(music)
        eeg_features = torch.from_numpy(self.predict(eeg.detach().cpu().numpy())).to(self.device)
        loss, acc = self.loss_fn(eeg_features, music_features, return_acc=True)
        return ModelOutput(
            loss = loss,
            acc = acc,
            eeg_features = eeg_features,
            music_features = music_features
        )

class WhisperEncoderFourier(WhisperEncoder):
    def __init__(self, model_name, aligned_space_dim=512, linear=True):
        super(WhisperEncoderFourier, self).__init__(model_name, aligned_space_dim, linear)
        # self.aligned_space_proj = nn.Linear(self.hidden_size, self.aligned_space_dim).to(torch.cfloat)
        # self.cnn_fourier = nn.Sequential(
        #     LinearFourier(self.hidden_size, self.hidden_size, activation=nn.GELU()),
        #     LinearFourier(self.hidden_size, self.hidden_size, activation=nn.GELU()),
        # )
        self.aligned_space_proj = LinearFourier(self.hidden_size, self.aligned_space_dim)
        # self.aligned_space_proj = nn.Sequential(
        #     LinearFourier(self.hidden_size * 2, self.hidden_size * 2, activation=nn.GELU(), dropout=0.0),
        #     nn.GELU(),
        #     nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
        #     nn.GELU(),
        #     nn.Linear(self.hidden_size * 2, self.aligned_space_dim)
        # )

    @torch.no_grad()
    def preprocess(self, x):
        if isinstance(x, torch.Tensor):
            audio = x.cpu().numpy()
        else:
            audio = x
        inputs = self.feature_extractor(audio, return_tensors="pt", sampling_rate=16000).input_features
        return inputs

    @torch.no_grad()
    def forward_whisper(self, x):
        return self.model(x, output_hidden_states=True).hidden_states[-1]

    def forward(self, x):
        audio_features = self.forward_whisper(x) # [batch_size, seq_len, hidden_size]
        audio_features = torch.fft.fft(audio_features, dim=1)
        # audio_features = self.cnn_fourier(audio_features)
        audio_features = audio_features.mean(dim=-2)
        audio_features = self.aligned_space_proj(audio_features)
        return audio_features