#!/usr/bin/env python3
"""Re-evaluate merged checkpoints and prune to the top-K by pass@32."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


STEP_RE = re.compile(r"^global_step_(\d+)$")


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parents[2]
    default_ckpt_root = repo_dir / "output/ckpt_best3_selection_on/ckpt"

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing merged global_step_* models, write checkpoint eval reports, "
            "rank by pass@32, and optionally prune non-top checkpoints."
        )
    )
    parser.add_argument("--ckpt-root", type=Path, default=default_ckpt_root)
    parser.add_argument("--merged-root", type=Path, default=None)
    parser.add_argument("--original-root", type=Path, default=None)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--eval-script", type=Path, default=repo_dir / "eval/openonerec_eval.py")
    parser.add_argument("--server-script", type=Path, default=repo_dir / "recipe/onerec/vllm_openai_server_compat.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--test-parquet", type=Path, default=repo_dir.parent / "data/test.parquet")
    parser.add_argument("--test-max-sample", type=int, default=int(os.environ.get("EVAL_TEST_MAX_SAMPLE", "-1")))
    parser.add_argument("--k-values", default=os.environ.get("EVAL_K_VALUES", "1,32"))
    parser.add_argument("--metric", default="pass@32")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--backend", choices=("offline", "serving"), default=os.environ.get("EVAL_BACKEND", "offline"))
    parser.add_argument("--server-host", default=os.environ.get("EVAL_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--server-base-port", type=int, default=int(os.environ.get("EVAL_SERVER_BASE_PORT", "18000")))
    parser.add_argument("--server-start-timeout", type=int, default=int(os.environ.get("EVAL_SERVER_START_TIMEOUT", "600")))
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=int(os.environ.get("EVAL_TENSOR_PARALLEL_SIZE", "1")),
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=int(os.environ.get("EVAL_DATA_PARALLEL_SIZE", "8")),
        help="Number of concurrent single-GPU checkpoint eval workers.",
    )
    parser.add_argument(
        "--eval-data-parallel-size",
        type=int,
        default=int(os.environ.get("EVAL_INTERNAL_DATA_PARALLEL_SIZE", "1")),
        help="vLLM internal data_parallel_size for each eval worker; keep at 1 for local recovery.",
    )
    parser.add_argument("--cuda-devices", default=os.environ.get("EVAL_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--steps", type=int, nargs="*", default=None, help="Only evaluate these global steps.")
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-run evaluation even when an existing JSON or log already has the metric.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--prune", action="store_true", help="Delete non-top-K merged and original checkpoints.")
    parser.add_argument(
        "--allow-partial-prune",
        action="store_true",
        help="Allow pruning even if some checkpoint evaluations failed.",
    )
    parser.add_argument(
        "--extra-eval-arg",
        action="append",
        default=[],
        help="Extra argument passed through to openonerec_eval.py. Repeat for multiple args.",
    )
    return parser.parse_args()


def step_from_path(path: Path) -> int:
    match = STEP_RE.match(path.name)
    if match is None:
        raise ValueError(f"Not a global_step directory: {path}")
    return int(match.group(1))


def discover_merged_checkpoints(merged_root: Path, selected_steps: set[int] | None) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in merged_root.glob("global_step_*"):
        if not path.is_dir():
            continue
        try:
            step = step_from_path(path)
        except ValueError:
            continue
        if selected_steps is not None and step not in selected_steps:
            continue
        checkpoints.append((step, path.resolve()))
    return sorted(checkpoints)


def read_metric(result_path: Path, metric: str) -> float | None:
    if not result_path.is_file():
        return None
    try:
        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    except Exception as exc:
        print(f"[recover_eval] Failed to read {result_path}: {exc}", flush=True)
        return None

    evaluation = result.get("evaluation", {})
    try:
        evaluated_samples = int(evaluation.get("evaluated_samples", 0))
        total_samples = int(evaluation.get("total_samples", 0))
    except (TypeError, ValueError):
        return None
    if evaluated_samples <= 0 or total_samples <= 0:
        return None

    value = evaluation.get(metric)
    if value is None and metric == "pass@32":
        value = evaluation.get("pass_at_32")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_metric_from_log(log_path: Path, metric: str) -> float | None:
    if not log_path.is_file():
        return None
    escaped_metric = re.escape(metric)
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){escaped_metric}\s*[:=]\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))")
    evaluated_pattern = re.compile(r"Evaluated\s*/\s*Total\s*:\s*(\d+)\s*/\s*(\d+)")
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"[recover_eval] Failed to read {log_path}: {exc}", flush=True)
        return None
    evaluated_matches = evaluated_pattern.findall(text)
    if evaluated_matches:
        evaluated, total = evaluated_matches[-1]
        if int(evaluated) <= 0 or int(total) <= 0:
            return None
    matches = pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def read_existing_metric(result_path: Path, log_path: Path, metric: str) -> tuple[float | None, str | None]:
    json_score = read_metric(result_path, metric)
    if json_score is not None:
        return json_score, str(result_path)

    log_score = parse_metric_from_log(log_path, metric)
    if log_score is not None:
        return log_score, str(log_path)

    return None, None


def parse_cuda_devices(cuda_devices: str) -> list[str]:
    return [device.strip() for device in cuda_devices.split(",") if device.strip()]


def build_worker_device_slots(args: argparse.Namespace) -> list[str | None]:
    worker_count = max(1, int(args.data_parallel_size))
    devices = parse_cuda_devices(args.cuda_devices)
    devices_per_worker = max(1, int(args.tensor_parallel_size) * int(args.eval_data_parallel_size))
    if not devices:
        if worker_count > 1:
            raise ValueError("--cuda-devices or CUDA_VISIBLE_DEVICES must be set when using multiple eval workers")
        return [None]
    required_devices = worker_count * devices_per_worker
    if len(devices) < required_devices:
        raise ValueError(
            f"Need {required_devices} CUDA devices for {worker_count} workers with "
            f"{devices_per_worker} device(s) each, but got: {','.join(devices)}"
        )
    return [
        ",".join(devices[index * devices_per_worker : (index + 1) * devices_per_worker])
        for index in range(worker_count)
    ]


def wait_for_server(host: str, port: int, process: subprocess.Popen, timeout: int) -> bool:
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
        except URLError:
            pass
        except TimeoutError:
            pass
        time.sleep(2)
    return False


def stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            process.kill()
        process.wait(timeout=30)


def start_vllm_server(
    args: argparse.Namespace,
    merged_model: Path,
    worker_id: int | None,
    cuda_devices: str | None,
    server_log_path: Path,
) -> tuple[subprocess.Popen, int]:
    if worker_id is None:
        worker_id = 0
    port = int(args.server_base_port) + worker_id
    cmd = [
        args.python,
        str(args.server_script),
        "--model",
        str(merged_model),
        "--served-model-name",
        str(merged_model),
        "--host",
        args.server_host,
        "--port",
        str(port),
        "--trust-remote-code",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enforce-eager",
    ]
    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])

    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        env["EVAL_CUDA_VISIBLE_DEVICES"] = cuda_devices

    print(
        f"[recover_eval] Starting vLLM server worker={worker_id} cuda={cuda_devices} "
        f"port={port} model={merged_model}",
        flush=True,
    )
    server_log = server_log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(args.eval_script.resolve().parents[1]),
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    finally:
        server_log.close()

    if not wait_for_server(args.server_host, port, process, args.server_start_timeout):
        stop_process_group(process)
        raise RuntimeError(f"vLLM server failed to become ready on {args.server_host}:{port}; see {server_log_path}")
    print(f"[recover_eval] vLLM server ready on {args.server_host}:{port}", flush=True)
    return process, port


def run_eval(
    args: argparse.Namespace,
    step: int,
    merged_model: Path,
    result_dir: Path,
    worker_id: int | None = None,
    cuda_devices: str | None = None,
) -> tuple[float | None, Path, Path]:
    result_path = result_dir / f"global_step_{step}.json"
    log_path = result_dir / f"global_step_{step}.log"

    if not args.force_eval:
        existing_score, existing_source = read_existing_metric(result_path, log_path, args.metric)
        if existing_score is not None:
            print(
                f"[recover_eval] Reusing global_step_{step}: "
                f"{args.metric}={existing_score:.6f} from {existing_source}",
                flush=True,
            )
            return existing_score, result_path, log_path

    cmd = [
        args.python,
        str(args.eval_script),
        "--model-path",
        str(merged_model),
        "--backend",
        args.backend,
        "--test-max-sample",
        str(args.test_max_sample),
        "--trust-remote-code",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--data-parallel-size",
        str(args.eval_data_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enforce-eager",
        "--test-parquet",
        str(args.test_parquet),
        "--k-values",
        args.k_values,
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_path.name,
    ]
    if args.backend == "serving":
        port = int(args.server_base_port) + (worker_id or 0)
        cmd.extend(["--host", args.server_host, "--port", str(port)])
    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    cmd.extend(args.extra_eval_arg)

    worker_prefix = f"worker={worker_id} cuda={cuda_devices}" if worker_id is not None else "worker=main"
    print(f"[recover_eval] Evaluating global_step_{step} ({worker_prefix}): {' '.join(cmd)}", flush=True)
    started_at = time.time()
    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        env["EVAL_CUDA_VISIBLE_DEVICES"] = cuda_devices
    server_process = None
    server_log_path = log_path.with_name(f"{log_path.stem}.server.log")
    if args.backend == "serving":
        try:
            server_process, _ = start_vllm_server(args, merged_model, worker_id, cuda_devices, server_log_path)
        except Exception as exc:
            log_path.write_text(f"[recover_eval] Failed to start vLLM server: {exc}\n", encoding="utf-8")
            print(f"[recover_eval] Failed to start server for global_step_{step}: {exc}", flush=True)
            return None, result_path, log_path
    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(args.eval_script.resolve().parents[1]),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while True:
                returncode = process.poll()
                if returncode is not None:
                    break
                time.sleep(60)
                elapsed = time.time() - started_at
                print(
                    f"[recover_eval] global_step_{step} ({worker_prefix}) still running after {elapsed:.0f}s; log={log_path}",
                    flush=True,
                )
        finally:
            if server_process is not None:
                stop_process_group(server_process)

    elapsed = time.time() - started_at
    print(
        f"[recover_eval] global_step_{step} exit={returncode} elapsed={elapsed:.1f}s log={log_path}",
        flush=True,
    )
    if returncode != 0:
        try:
            output_tail = log_path.read_text(encoding="utf-8")[-4000:]
        except Exception:
            output_tail = ""
        if output_tail:
            print(output_tail, flush=True)
        return None, result_path, log_path

    score = read_metric(result_path, args.metric)
    if score is None:
        print(f"[recover_eval] Could not parse {args.metric} from {result_path}", flush=True)
    else:
        print(f"[recover_eval] global_step_{step} {args.metric}={score:.6f}", flush=True)
    return score, result_path, log_path


def evaluate_checkpoints(
    args: argparse.Namespace,
    checkpoints: list[tuple[int, Path]],
    result_dir: Path,
    original_root: Path,
) -> tuple[list[dict], list[int]]:
    if args.data_parallel_size <= 1:
        records = []
        failed_steps = []
        for step, merged_model in checkpoints:
            score, result_path, log_path = run_eval(args, step, merged_model, result_dir)
            if score is None:
                failed_steps.append(step)
                continue
            records.append(make_record(args, step, score, merged_model, result_path, log_path, original_root))
        return records, failed_steps

    device_slots = build_worker_device_slots(args)
    work_queue: queue.Queue[tuple[int, Path]] = queue.Queue()
    for checkpoint in checkpoints:
        work_queue.put(checkpoint)

    records = []
    failed_steps = []
    lock = threading.Lock()

    def worker(worker_id: int, cuda_devices: str | None) -> None:
        while True:
            try:
                step, merged_model = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                score, result_path, log_path = run_eval(
                    args,
                    step,
                    merged_model,
                    result_dir,
                    worker_id=worker_id,
                    cuda_devices=cuda_devices,
                )
                with lock:
                    if score is None:
                        failed_steps.append(step)
                    else:
                        records.append(make_record(args, step, score, merged_model, result_path, log_path, original_root))
            finally:
                work_queue.task_done()

    print(
        "[recover_eval] Launching "
        f"{len(device_slots)} concurrent TP={args.tensor_parallel_size} eval workers on CUDA slots: "
        + ", ".join(slot or "<inherited>" for slot in device_slots),
        flush=True,
    )
    threads = [
        threading.Thread(target=worker, args=(worker_id, cuda_devices), daemon=True)
        for worker_id, cuda_devices in enumerate(device_slots)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return records, failed_steps


def make_record(
    args: argparse.Namespace,
    step: int,
    score: float,
    merged_model: Path,
    result_path: Path,
    log_path: Path,
    original_root: Path,
) -> dict:
    original_path = (original_root / f"global_step_{step}").resolve()
    return {
        "score": float(score),
        "metric": args.metric,
        "path": str(original_path),
        "merged_model_dir": str(merged_model),
        "result_path": str(result_path.resolve()),
        "log_path": str(log_path.resolve()),
        "global_step": int(step),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def split_valid_json_prefix(
    args: argparse.Namespace,
    checkpoints: list[tuple[int, Path]],
    result_dir: Path,
    original_root: Path,
) -> tuple[list[dict], list[tuple[int, Path]]]:
    if args.force_eval:
        return [], checkpoints

    prefix_records = []
    for index, (step, merged_model) in enumerate(checkpoints):
        result_path = result_dir / f"global_step_{step}.json"
        log_path = result_dir / f"global_step_{step}.log"
        score = read_metric(result_path, args.metric)
        if score is None:
            print(
                f"[recover_eval] First checkpoint without valid {args.metric} JSON: "
                f"global_step_{step}; starting evaluation from this step.",
                flush=True,
            )
            return prefix_records, checkpoints[index:]
        prefix_records.append(make_record(args, step, score, merged_model, result_path, log_path, original_root))

    print(
        f"[recover_eval] All {len(checkpoints)} checkpoints already have valid {args.metric} JSON.",
        flush=True,
    )
    return prefix_records, []


def write_ranking(
    ckpt_root: Path,
    records: list[dict],
    keep_records: list[dict],
    metric: str,
    top_k: int,
) -> None:
    ranking_path = ckpt_root / "checkpoint_eval_recovery_ranking.json"
    best_path = ckpt_root / "best_pass32_checkpoints.json"
    payload = {
        "metric": metric,
        "keep_count": top_k,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoints": keep_records,
        "all_evaluated_checkpoints": records,
    }
    ranking_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    best_path.write_text(
        json.dumps({"metric": metric, "keep_count": top_k, "checkpoints": keep_records}, indent=2),
        encoding="utf-8",
    )
    print(f"[recover_eval] Wrote ranking: {ranking_path}", flush=True)
    print(f"[recover_eval] Wrote trainer-compatible best file: {best_path}", flush=True)


def remove_tree(path: Path) -> None:
    if path.is_dir():
        print(f"[recover_eval] Removing {path}", flush=True)
        shutil.rmtree(path)


def main() -> int:
    args = parse_args()
    args.ckpt_root = args.ckpt_root.expanduser().resolve()
    merged_root = (args.merged_root or args.ckpt_root / "checkpoint_eval_merged").expanduser().resolve()
    original_root = (args.original_root or args.ckpt_root).expanduser().resolve()
    result_dir = (args.result_dir or args.ckpt_root / "checkpoint_eval_results").expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if not merged_root.is_dir():
        raise FileNotFoundError(f"Merged checkpoint root does not exist: {merged_root}")
    if not args.eval_script.is_file():
        raise FileNotFoundError(f"Eval script does not exist: {args.eval_script}")
    if args.backend == "serving" and not args.server_script.is_file():
        raise FileNotFoundError(f"vLLM server launcher does not exist: {args.server_script}")
    if not args.test_parquet.is_file():
        raise FileNotFoundError(f"Test parquet does not exist: {args.test_parquet}")

    selected_steps = set(args.steps) if args.steps is not None else None
    checkpoints = discover_merged_checkpoints(merged_root, selected_steps)
    if not checkpoints:
        raise FileNotFoundError(f"No merged global_step_* checkpoints found under {merged_root}")

    print(f"[recover_eval] Found {len(checkpoints)} merged checkpoints under {merged_root}", flush=True)
    prefix_records, checkpoints_to_eval = split_valid_json_prefix(args, checkpoints, result_dir, original_root)
    if prefix_records:
        first_step = prefix_records[0]["global_step"]
        last_step = prefix_records[-1]["global_step"]
        print(
            f"[recover_eval] Reusing contiguous valid JSON prefix: "
            f"global_step_{first_step}..global_step_{last_step} ({len(prefix_records)} checkpoints).",
            flush=True,
        )

    records, failed_steps = evaluate_checkpoints(args, checkpoints_to_eval, result_dir, original_root)
    records.extend(prefix_records)

    if not records:
        print("[recover_eval] No successful evaluations; refusing to prune.", flush=True)
        return 1

    records.sort(key=lambda record: (float(record["score"]), int(record["global_step"])), reverse=True)
    keep_records = records[: args.top_k]
    keep_steps = {int(record["global_step"]) for record in keep_records}
    write_ranking(args.ckpt_root, records, keep_records, args.metric, args.top_k)

    print("[recover_eval] Top checkpoints:", flush=True)
    for rank, record in enumerate(keep_records, start=1):
        print(
            f"  #{rank}: global_step_{record['global_step']} {args.metric}={float(record['score']):.6f}",
            flush=True,
        )

    if failed_steps:
        print(f"[recover_eval] Failed steps: {failed_steps}", flush=True)
        if args.prune and not args.allow_partial_prune:
            print("[recover_eval] Refusing to prune because some evaluations failed. Use --allow-partial-prune to override.", flush=True)
            return 2

    if not args.prune:
        print("[recover_eval] Prune not requested. Re-run with --prune to delete non-top-K checkpoints.", flush=True)
        return 0

    all_successful_steps = {int(record["global_step"]) for record in records}
    removable_steps = all_successful_steps - keep_steps
    for step in sorted(removable_steps):
        remove_tree(merged_root / f"global_step_{step}")
        remove_tree(original_root / f"global_step_{step}")

    latest_tracker = args.ckpt_root / "latest_checkpointed_iteration.txt"
    latest_tracker.write_text(str(max(keep_steps)), encoding="utf-8")
    print(f"[recover_eval] Updated {latest_tracker} to {max(keep_steps)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
