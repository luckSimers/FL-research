# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .wrn import wrn_28_2, wrn_28_8, wrn_var_37_2
from .vit import vit_base_patch16_224, vit_small_patch16_224, vit_small_patch2_32, vit_tiny_patch2_32, vit_base_patch16_96
from .bert import bert_base_cased, bert_base_uncased
from .wave2vecv2 import wave2vecv2_base
from .hubert import hubert_base
from .simple_cnn import ModerateCNN
from .fl_vae import VAE