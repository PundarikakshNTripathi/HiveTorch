"""Model-state (de)serialisation for network transport: safetensors bytes, no pickle."""
from collections import OrderedDict
from typing import Dict

import torch
from safetensors.torch import load as _load
from safetensors.torch import save as _save


def state_to_bytes(state: Dict[str, torch.Tensor]) -> bytes:
    """Serialise a state dict. safetensors is a flat, typed, pickle-free format, so a malicious peer cannot
    execute code by sending crafted weights."""
    return _save({k: v.detach().contiguous().cpu() for k, v in state.items()})


def bytes_to_state(payload: bytes) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict(_load(payload))
