# Copyright 2025 OpenOneRec Contributors
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

"""
Registration module for MultiRewardManager into verl's reward manager registry.

Usage:
    # Import this module before training to register the "multi" reward manager.
    # This can be done in a custom entrypoint or via the config's
    # custom_reward_function mechanism.

    import multireward_mgr_support.reward_manager.register_multi  # noqa: F401

    # Then in your verl config:
    #   reward_model:
    #     reward_manager: multi
"""

from verl.workers.reward_manager import register
from .multi_reward_manager import MultiRewardManager

# Register under the name "multi" so it can be referenced in verl configs
register("multi")(MultiRewardManager)
