#!/usr/bin/env python3
"""Run the OpenOneRec evaluator with this checkout as its project root."""

from pathlib import Path


BACKUP_EVAL = Path(
    "/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/verl-gr-fork-workingbranch/eval/openonerec_eval.py"
)


def _ensure_tokenizer_compatibility() -> None:
    """Patch tokenizer APIs expected by this vLLM build before eval imports vLLM."""
    try:
        from transformers import PreTrainedTokenizerBase
    except ImportError:
        return

    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return

    @property
    def all_special_tokens_extended(self):
        special_map = getattr(self, "special_tokens_map_extended", None) or {}
        tokens = []
        for value in special_map.values():
            if isinstance(value, (list, tuple)):
                tokens.extend(value)
            elif value is not None:
                tokens.append(value)
        if tokens:
            return tokens
        return list(getattr(self, "all_special_tokens", []))

    PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended


def _add_data_parallel_eval_support(source: str) -> str:
    """Add local DP CLI plumbing to the borrowed evaluator when it is missing."""
    if "--data-parallel-size" in source:
        return source

    source = source.replace(
        '    infer.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)\n',
        '    infer.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)\n'
        '    infer.add_argument(\n'
        '        "--data-parallel-size",\n'
        '        "--dp",\n'
        '        type=int,\n'
        '        default=int(os.environ.get("EVAL_INTERNAL_DATA_PARALLEL_SIZE", "1")),\n'
        '    )\n',
    )
    source = source.replace(
        '            "tensor_parallel_size": args.tensor_parallel_size,\n',
        '            "tensor_parallel_size": args.tensor_parallel_size,\n'
        '            "data_parallel_size": args.data_parallel_size,\n',
    )
    return source


def main() -> None:
    if not BACKUP_EVAL.is_file():
        raise FileNotFoundError(f"Backup evaluator not found: {BACKUP_EVAL}")
    _ensure_tokenizer_compatibility()
    globals_dict = {
        "__file__": __file__,
        "__name__": "__main__",
        "__package__": None,
        "__builtins__": __builtins__,
    }
    source = BACKUP_EVAL.read_text(encoding="utf-8")
    source = _add_data_parallel_eval_support(source)
    exec(compile(source, str(BACKUP_EVAL), "exec"), globals_dict)


if __name__ == "__main__":
    main()
