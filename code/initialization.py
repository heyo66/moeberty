# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2023 OLMo Authors
# License: Apache-2.0
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# License: Apache-2.0

import math
from enum import Enum
from typing import Optional

import torch.nn as nn

__all__ = ["init_weights", "ModuleType", "InitFnType"]


class InitFnType(str, Enum):
    full_megatron = "full_megatron"
    """
    This is what metaseq calls "full megatron init". It is the init used for Llama 2.
    """


class ModuleType(str, Enum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config,
    module: nn.Module,
    layer_dim: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    if getattr(config, "init_method", InitFnType.full_megatron) not in (
        InitFnType.full_megatron,
        InitFnType.full_megatron.value,
    ):
        raise NotImplementedError(getattr(config, "init_method", None))

    if getattr(config, "init_small_embedding", False):
        raise ValueError("Cannot use 'small_embedding_init' with 'full_megatron' init.")

    if type_of_module is None:
        raise RuntimeError(f"When using the {InitFnType.full_megatron} init, every module must have a type.")

    cutoff_factor = getattr(config, "init_cutoff_factor", None)
    if cutoff_factor is None:
        cutoff_factor = 3

    if type_of_module == ModuleType.in_module:
        std = config.init_std
    elif type_of_module == ModuleType.out_module:
        std = config.init_std / math.sqrt(2.0 * config.num_hidden_layers)
    elif type_of_module == ModuleType.emb:
        std = config.init_std
    elif type_of_module == ModuleType.final_out:
        std = config.hidden_size**-0.5
    else:
        raise RuntimeError(f"Unknown module type '{type_of_module}'")

    nn.init.trunc_normal_(
        module.weight,
        mean=0.0,
        std=std,
        a=-cutoff_factor * std,
        b=cutoff_factor * std,
    )

    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)
