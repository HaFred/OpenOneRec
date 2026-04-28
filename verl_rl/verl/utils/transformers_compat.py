# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Compatibility helpers for transformers auto model classes."""


def get_auto_model_for_vision2seq():
    """Return a compatible auto model class for vision-to-sequence generation."""
    try:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq
    except ImportError:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText


def is_vision2seq_config(model_config) -> bool:
    """Whether the config is supported by the vision-to-sequence auto class."""
    auto_model_cls = get_auto_model_for_vision2seq()
    model_mapping = getattr(auto_model_cls, "_model_mapping", None)
    if model_mapping is None:
        return False
    return type(model_config) in model_mapping.keys()
