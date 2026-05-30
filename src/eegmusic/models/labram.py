import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PreTrainedModel, PretrainedConfig

from .transformer import Transformer
from .common import ModelOutput
from .whisper import WhisperEncoder, CLAPEncoder
from ..utils.mask import random_masking
from ..labram import (
    vqnsp_encoder_base_decoder_3x200x12,
    vqnsp_encoder_large_decoder_3x200x24,
)

_TOKENIZER_FACTORY = {
    "vqnsp_encoder_base_decoder_3x200x12": vqnsp_encoder_base_decoder_3x200x12,
    "vqnsp_encoder_large_decoder_3x200x24": vqnsp_encoder_large_decoder_3x200x24,
}


class LaBraMConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super(LaBraMConfig, self).__init__(**kwargs)
        self.embed_dim = kwargs.get("embed_dim", 768)
        self.patch_size = kwargs.get("patch_size", 100)
        self.eeg_length = kwargs.get("eeg_length", 1600)
        self.num_channels = kwargs.get("num_channels", 125)
        self.aligned_space_dim = kwargs.get("aligned_space_dim", 1024)
        self.num_heads = kwargs.get("num_heads", 12)
        self.num_layers = kwargs.get("num_layers", 12)
        self.dropout = kwargs.get("dropout", 0.0)

        self.decoder_embed_dim = kwargs.get("decoder_embed_dim", 512)
        self.decoder_num_heads = kwargs.get("decoder_num_heads", 8)
        self.decoder_num_layers = kwargs.get("decoder_num_layers", 6)

        self.mask_ratio = kwargs.get("mask_ratio", 0.75)
        self.use_tokenizer = kwargs.get("use_tokenizer", False)
        self.tokenizer_model = kwargs.get("tokenizer_model", "vqnsp_encoder_base_decoder_3x200x12")
        self.tokenizer_weight = kwargs.get("tokenizer_weight", None)


class LaBraMForPretraining(PreTrainedModel):
    """
    LaBraM-style masked modeling for EEG channel patches.
    """
    supports_gradient_checkpointing = True
    config_class = LaBraMConfig

    def __init__(self, config: LaBraMConfig):
        super(LaBraMForPretraining, self).__init__(config)
        self.use_tokenizer = config.use_tokenizer
        self.tokenizer = None
        self.tokenizer_patch_size = None
        self.codebook_size = None

        if self.use_tokenizer:
            if not config.tokenizer_weight:
                raise ValueError("tokenizer_weight must be set when use_tokenizer=True.")
            tokenizer_builder = _TOKENIZER_FACTORY.get(config.tokenizer_model)
            if tokenizer_builder is None:
                raise ValueError(f"Unsupported tokenizer_model: {config.tokenizer_model}")
            self.tokenizer = tokenizer_builder(
                pretrained=True,
                pretrained_weight=config.tokenizer_weight,
                as_tokenzer=True,
                EEG_size=config.eeg_length,
                patch_size=config.patch_size,
                num_channels=config.num_channels,
            )
            self.tokenizer.eval()
            self.tokenizer.requires_grad_(False)
            self.tokenizer_patch_size = int(self.tokenizer.patch_size)
            if config.eeg_length % self.tokenizer_patch_size != 0:
                raise ValueError("eeg_length must be divisible by tokenizer patch_size.")
            self.num_patches = config.eeg_length // self.tokenizer_patch_size
        else:
            if config.eeg_length % config.patch_size != 0:
                raise ValueError("eeg_length must be divisible by patch_size.")
            self.num_patches = config.eeg_length // config.patch_size

        self.num_tokens = config.num_channels * self.num_patches

        if not self.use_tokenizer:
            self.patch_embed = nn.Conv2d(
                1,
                config.embed_dim,
                kernel_size=(1, config.patch_size),
                stride=(1, config.patch_size),
            )
            self.pos_embed = nn.Parameter(
                torch.zeros(1, config.num_channels, self.num_patches, config.embed_dim),
                requires_grad=True,
            )
        else:
            self.patch_embed = None
            self.pos_embed = None
            self.token_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_tokens, config.embed_dim),
                requires_grad=True,
            )
            self.codebook_size = int(self.tokenizer.get_number_of_tokens())
            self.token_embed = nn.Embedding(self.codebook_size, config.embed_dim)
        self.encoder = Transformer(
            emb_dim=config.embed_dim,
            num_heads=config.num_heads,
            ff_dim=config.embed_dim * 4,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.encoder_norm = nn.LayerNorm(config.embed_dim)

        self.decoder_embed = nn.Linear(config.embed_dim, config.decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, config.decoder_embed_dim),
            requires_grad=True,
        )
        self.decoder = Transformer(
            emb_dim=config.decoder_embed_dim,
            num_heads=config.decoder_num_heads,
            ff_dim=config.decoder_embed_dim * 4,
            num_layers=config.decoder_num_layers,
            dropout=config.dropout,
        )
        self.decoder_norm = nn.LayerNorm(config.decoder_embed_dim)
        out_dim = self.codebook_size if self.use_tokenizer else config.patch_size
        self.decoder_pred = nn.Linear(config.decoder_embed_dim, out_dim)

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.use_tokenizer:
            nn.init.trunc_normal_(self.token_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _resize_eeg(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_channels, ts_len]
        target_len = self.config.eeg_length
        if x.shape[-1] < target_len:
            pad_len = target_len - x.shape[-1]
            x = F.pad(x, (0, pad_len))
        elif x.shape[-1] > target_len:
            x = x[..., :target_len]
        return x

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_channels, eeg_length]
        # return: [batch_size, num_channels, num_patches, patch_size]
        x = self._resize_eeg(x)
        b, c, t = x.shape
        x = x.reshape(b, c, self.num_patches, self.config.patch_size)
        return x

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float):
        # x: [batch_size, num_channels, eeg_length]
        if self.use_tokenizer:
            raise RuntimeError("forward_encoder is not used when use_tokenizer=True.")
        x = self._resize_eeg(x)
        x = x.unsqueeze(1)  # [batch_size, 1, num_channels, eeg_length]
        x = self.patch_embed(x)  # [batch_size, embed_dim, num_channels, num_patches]
        x = rearrange(x, "b c h w -> b h w c")  # [batch_size, num_channels, num_patches, embed_dim]
        x = x + self.pos_embed
        x = rearrange(x, "b h w c -> b (h w) c")  # [batch_size, num_tokens, embed_dim]

        mask, ids_restore, ids_keep = random_masking(x.shape[1], x.shape[0], mask_ratio)
        x = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]).to(x.device),
        )
        x = self.encoder(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_channels, eeg_length]
        # token_ids: [batch_size, num_tokens]
        x = self._resize_eeg(x)
        x = rearrange(
            x,
            "b c (a t) -> b c a t",
            t=self.tokenizer_patch_size,
        )
        input_chans = torch.arange(self.config.num_channels + 1, device=x.device)
        with torch.no_grad():
            token_ids = self.tokenizer.get_codebook_indices(x, input_chans=input_chans)
        return token_ids.to(x.device)

    def forward(self, x: torch.Tensor, mask_ratio: float = None):
        """
        x: [batch_size, num_channels, eeg_length]
        pred (regression): [batch_size, num_tokens, patch_size]
        pred (tokenizer): [batch_size, num_tokens, codebook_size]
        """
        if mask_ratio is None:
            mask_ratio = self.config.mask_ratio

        if self.use_tokenizer:
            # tokenizer path
            token_ids = self.tokenize(x)
            x = self.token_embed(token_ids)  # [batch_size, num_tokens, embed_dim]
            x = x + self.token_pos_embed

            mask, ids_restore, ids_keep = random_masking(x.shape[1], x.shape[0], mask_ratio)
            x = torch.gather(
                x,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]).to(x.device),
            )
            x = self.encoder(x)
            x = self.encoder_norm(x)

            dec = self.decoder_embed(x)
            mask_tokens = self.mask_token.repeat(dec.shape[0], ids_restore.shape[1] - dec.shape[1], 1)
            dec = torch.cat([dec, mask_tokens], dim=1)
            dec = torch.gather(
                dec,
                dim=1,
                index=ids_restore.unsqueeze(-1).repeat(1, 1, dec.shape[-1]).to(dec.device),
            )
            dec = dec + self.decoder_pos_embed
            dec = self.decoder(dec)
            dec = self.decoder_norm(dec)
            pred_logits = self.decoder_pred(dec)  # [batch_size, num_tokens, codebook_size]

            loss = F.cross_entropy(
                pred_logits.reshape(-1, self.codebook_size),
                token_ids.reshape(-1),
                reduction="none",
            )
            loss = loss.view(token_ids.shape[0], -1)
            loss = (loss * mask.to(loss.device)).sum() / mask.sum().clamp(min=1)

            return ModelOutput(
                loss=loss,
                last_hidden_state=x,
                hidden_states=pred_logits,
            )

        patches = self.patchify(x)
        target = rearrange(patches, "b c n p -> b (c n) p")  # [batch_size, num_tokens, patch_size]

        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)

        # decoder
        dec = self.decoder_embed(latent)
        mask_tokens = self.mask_token.repeat(dec.shape[0], ids_restore.shape[1] - dec.shape[1], 1)
        dec = torch.cat([dec, mask_tokens], dim=1)
        dec = torch.gather(
            dec,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, dec.shape[-1]).to(dec.device),
        )
        dec = dec + self.decoder_pos_embed
        dec = self.decoder(dec)
        dec = self.decoder_norm(dec)
        pred = self.decoder_pred(dec)  # [batch_size, num_tokens, patch_size]

        # compute loss on masked patches
        loss = (pred - target).pow(2).mean(dim=-1)
        loss = (loss * mask.to(loss.device)).sum() / mask.sum().clamp(min=1)

        return ModelOutput(
            loss=loss,
            last_hidden_state=latent,
            hidden_states=pred,
        )


class LaBraMForAlign(LaBraMForPretraining):
    """
    EEG-music alignment model based on LaBraM encoder.
    """
    def __init__(self, config: LaBraMConfig):
        super(LaBraMForAlign, self).__init__(config)
        self.mlp = nn.Linear(config.embed_dim, config.embed_dim)
        self.aligned_space_proj = nn.Linear(config.embed_dim, config.aligned_space_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.out_norm_aligned_space = nn.LayerNorm(config.aligned_space_dim)

        self.whisper = WhisperEncoder(
            model_name="openai/whisper-base",
            aligned_space_dim=config.aligned_space_dim,
        )

        nn.init.trunc_normal_(self.aligned_space_proj.weight, std=0.02)
        self.aligned_space_proj.bias.data.zero_()
        nn.init.trunc_normal_(self.mlp.weight, std=0.02)
        self.mlp.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, path):
        model = super().from_pretrained(path)
        model.whisper = WhisperEncoder(
            model_name="openai/whisper-base",
            aligned_space_dim=model.config.aligned_space_dim,
        )
        model.whisper.model.requires_grad = False
        return model
    
    @classmethod
    def from_pretrained_with_whisper(self, path):
        model = super().from_pretrained(path)
        return model
 
    def forward_encoder(self, x: torch.Tensor, channel_dropout: float = 0.0):
        # x: [batch_size, num_channels, eeg_length]
        if self.use_tokenizer:
            token_ids = self.tokenize(x)  # [batch_size, num_tokens]
            x = self.token_embed(token_ids)  # [batch_size, num_tokens, embed_dim]
            x = x + self.token_pos_embed
        else:
            x = self._resize_eeg(x)
            x = x.unsqueeze(1)  # [batch_size, 1, num_channels, eeg_length]
            x = self.patch_embed(x)  # [batch_size, embed_dim, num_channels, num_patches]
            x = rearrange(x, "b c h w -> b h w c")  # [batch_size, num_channels, num_patches, embed_dim]
            x = x + self.pos_embed
            x = rearrange(x, "b h w c -> b (h w) c")  # [batch_size, num_tokens, embed_dim]
        x = self.encoder(x)
        x = self.encoder_norm(x)
        return x

    def forward_eeg_encoder(self, x: torch.Tensor, channel_dropout: float = 0.0):
        # x: [batch_size, num_channels, eeg_length]
        x = self.forward_encoder(x, channel_dropout)
        # x: [batch_size, num_tokens, embed_dim]
        aligned_space = F.gelu(self.mlp(x.mean(dim=1)))
        aligned_space = self.aligned_space_proj(aligned_space)
        aligned_space = self.out_norm_aligned_space(aligned_space)
        return aligned_space

    def forward_whisper(self, x):
        x = self.whisper.preprocess(x).to(self.device)
        x = self.whisper(x)
        return x

    def loss_fn(self, eeg_features, audio_features, return_acc=False):
        # clip contrastive loss
        logit_scale = self.logit_scale.exp()
        eeg_features = F.normalize(eeg_features, p=2, dim=-1)
        audio_features = F.normalize(audio_features, p=2, dim=-1)
        logits_per_eeg = logit_scale * eeg_features @ audio_features.T
        logits_per_audio = logits_per_eeg.T
        loss = F.cross_entropy(logits_per_eeg, torch.arange(logits_per_eeg.shape[0]).to(self.device))
        loss += F.cross_entropy(logits_per_audio, torch.arange(logits_per_audio.shape[0]).to(self.device))
        if return_acc:
            acc = (logits_per_eeg.argmax(dim=-1) == torch.arange(logits_per_eeg.shape[0]).to(self.device)).float().mean()
            return loss / 2.0, acc
        return loss / 2.0

    def forward(self, eeg, music, music_prepocessed=None, channel_dropout: float = 0.0):
        # eeg: [batch_size, num_channels, eeg_length]
        # music: raw waveform batch
        eeg_features = self.forward_eeg_encoder(eeg, channel_dropout)
        if music_prepocessed is None:
            audio_features = self.forward_whisper(music)
        else:
            audio_features = self.whisper(music_prepocessed)
        loss, acc = self.loss_fn(eeg_features, audio_features, return_acc=True)
        return ModelOutput(loss=loss, acc=acc)
