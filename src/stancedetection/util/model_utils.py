from typing import List

import torch
from torch import nn

NON_BLOCKING_FIELDS = {"labels_mlm", "label", "labels", "labels_weights"}


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]


def batch_to_device(batch, device):
    return {
        k: v.to(device, non_blocking=(device == "cuda" and k in NON_BLOCKING_FIELDS))
        if isinstance(v, torch.Tensor)
        else v
        for k, v in batch.items()
    }


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


def freeze_layers(
    roberta: nn.Module,
    freeze_embeddings: bool = False,
    freeze_layer_ids: List[int] = None,
):
    if freeze_embeddings:
        freeze_module(roberta.embeddings)

    if freeze_layer_ids is not None:
        for id_ in freeze_layer_ids:
            freeze_module(roberta.encoder.layer[id_])
