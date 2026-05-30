from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelOutput:
    last_hidden_state: torch.Tensor = None
    hidden_states: torch.Tensor = None
    aligned_space: torch.Tensor = None
    attentions: torch.Tensor = None
    loss: torch.Tensor = None
    acc: torch.Tensor = None
    cross_loss: torch.Tensor = None
    eeg_autoloss: torch.Tensor = None
    music_autoloss: torch.Tensor = None
    eeg_features: torch.Tensor = None
    music_features: torch.Tensor = None