"""Pre-flight validation for Megatron-Core model architecture support.

Checks that the HuggingFace model at the given path uses an architecture
registered in verl's mcore registry before launching a potentially long
training job.
"""

import argparse
import json
import sys
from pathlib import Path

SUPPORTED_MCORE_ARCHITECTURES = {
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "DeepseekV3ForCausalLM",
    "MixtralForCausalLM",
    "Qwen2_5_VLForConditionalGeneration",
    "Llama4ForConditionalGeneration",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
}


def validate_model_architecture_for_megatron(model_path: str, expected_arch: str | None = None) -> str:
    """Validate that model_path contains a config.json with a supported architecture.

    Args:
        model_path: Path to the HuggingFace model directory.
        expected_arch: If provided, also assert the model matches this exact architecture.

    Returns:
        The detected architecture string.

    Raises:
        SystemExit: If validation fails.
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        print(f"[megatron_mcore_support] ERROR: config.json not found at {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    architectures = config.get("architectures", [])
    if not architectures:
        print(f"[megatron_mcore_support] ERROR: 'architectures' key missing or empty in {config_path}")
        sys.exit(1)

    arch = architectures[0]

    if arch not in SUPPORTED_MCORE_ARCHITECTURES:
        print(
            f"[megatron_mcore_support] ERROR: Architecture '{arch}' is not supported by verl mcore.\n"
            f"  Supported: {sorted(SUPPORTED_MCORE_ARCHITECTURES)}"
        )
        sys.exit(1)

    if expected_arch and arch != expected_arch:
        print(
            f"[megatron_mcore_support] WARNING: Expected architecture '{expected_arch}' "
            f"but model uses '{arch}'. Proceeding anyway since '{arch}' is supported."
        )

    print(f"[megatron_mcore_support] OK: architecture '{arch}' is supported by verl mcore.")
    return arch


def main():
    parser = argparse.ArgumentParser(description="Validate model architecture for Megatron-Core support")
    parser.add_argument("--model-path", required=True, help="Path to HuggingFace model directory")
    parser.add_argument("--expected-arch", default=None, help="Expected model architecture class name")
    args = parser.parse_args()

    validate_model_architecture_for_megatron(args.model_path, args.expected_arch)


if __name__ == "__main__":
    main()
