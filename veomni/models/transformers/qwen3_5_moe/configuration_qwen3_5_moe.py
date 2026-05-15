# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for Qwen3.6-35B-A3B MoE model.

Extends Qwen3_5Config with MoE-specific parameters.
"""

from typing import List, Optional

from veomni.models.transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config


class Qwen3_5MoeConfig(Qwen3_5Config):
    """Configuration for Qwen3.5/Qwen3.6 MoE models."""

    model_type = "qwen3_5_moe"

    def __init__(
        self,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        norm_topk_prob: bool = True,
        router_aux_loss_coef: float = 0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef
