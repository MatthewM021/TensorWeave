#!/usr/bin/env python3
"""Measure one Milestone-3 model variant in one isolated CPU process."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time

import psutil
import torch
from torch import Tensor

import tnlm_v3
from tnlm_v3.binding import (
    BindingArchitectureConfig,
    BindingModelConfig,
    RoutedBindingModel,
)
from tnlm_v3.compact_artifact import deserialize_compact_binding_model
from tnlm_v3.data import BindingModelInputs
from tnlm_v3.routing import CurriculumSchedule, RoutingMode
from tnlm_v3.truncation import model_state_fingerprint


_SCHEMA_VERSION = 1
_MAX_FIXTURE_BYTES = 8 * 1024 * 1024
_MAX_MODEL_CONFIG_BYTES = 64 * 1024
_MAX_MODEL_BYTES = 64 * 1024 * 1024
_MAX_ARCHITECTURE_CARDINALITY = 4096
_MAX_MODEL_DIMENSION = 65536
_MAX_MODEL_ELEMENTS = 16_000_000
_MAX_BATCH_SIZE = 256
_MAX_TIME_STEPS = 4096
_MAX_INPUT_ELEMENTS = 4096
_MAX_ITERATIONS = 1000
_MAX_THREADS = 64
_MIN_SAMPLE_PERIOD_MS = 0.1
_MAX_SAMPLE_PERIOD_MS = 1000.0
_HEX = frozenset("0123456789abcdef")
_INPUT_DTYPES = {
    "token_ids": torch.int64,
    "event_kinds": torch.int64,
    "primary_key_ids": torch.int64,
    "secondary_key_ids": torch.int64,
    "arguments": torch.int64,
    "valid_mask": torch.bool,
}
_MODEL_CONFIG_KEYS = {
    "task",
    "d_model",
    "cp_rank",
    "router_hidden_dim",
    "routing_mode",
    "curriculum_schedule",
    "curriculum_seed",
    "scale_feature_dim",
    "straight_through_route_surrogate",
}
_TASK_KEYS = {"num_surface_keys", "value_cardinality", "branches"}
_SCHEDULE_KEYS = {
    "start_step",
    "end_step",
    "start_probability",
    "end_probability",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_digest(value: str, name: str, length: int = 64) -> str:
    if len(value) != length or any(character not in _HEX for character in value):
        raise ValueError(f"{name} must be a lowercase {length}-character hex digest")
    return value


def _finite_json(value: object, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string JSON key at {path}")
            _finite_json(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_json(child, f"{path}[{index}]")
        return
    raise TypeError(f"unsupported JSON value {type(value).__name__} at {path}")


def _encoded_json(value: Mapping[str, object], *, pretty: bool) -> bytes:
    _finite_json(value)
    text = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2 if pretty else None,
        sort_keys=True,
        separators=None if pretty else (",", ":"),
    )
    return (text + "\n").encode("utf-8")


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    encoded = _encoded_json(value, pretty=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant {value!r} is forbidden")


def _strict_json(data: bytes, name: str, *, canonical: bool = False) -> object:
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_json_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"{name} must be strict UTF-8 JSON") from error
    _finite_json(value)
    if canonical:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if data not in (encoded, encoded + b"\n"):
            raise ValueError(f"{name} must use canonical JSON")
    return value


def _read_bounded(path: Path, limit: int, name: str) -> bytes:
    with path.open("rb") as handle:
        value = handle.read(limit + 1)
    if len(value) > limit:
        raise ValueError(f"{name} exceeds the {limit}-byte safety limit")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"invalid {name} fields; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _plain_int(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _parse_model_config(value: object) -> BindingModelConfig:
    document = _mapping(value, "model config")
    _exact_keys(document, _MODEL_CONFIG_KEYS, "model config")
    task_value = _mapping(document["task"], "model config task")
    _exact_keys(task_value, _TASK_KEYS, "model config task")
    task = BindingArchitectureConfig(
        num_surface_keys=_plain_int(
            task_value["num_surface_keys"],
            "num_surface_keys",
            minimum=2,
            maximum=_MAX_ARCHITECTURE_CARDINALITY,
        ),
        value_cardinality=_plain_int(
            task_value["value_cardinality"],
            "value_cardinality",
            minimum=2,
            maximum=_MAX_ARCHITECTURE_CARDINALITY,
        ),
        branches=_plain_int(
            task_value["branches"],
            "branches",
            minimum=2,
            maximum=_MAX_ARCHITECTURE_CARDINALITY,
        ),
    )

    schedule_value = document["curriculum_schedule"]
    schedule: CurriculumSchedule | None
    if schedule_value is None:
        schedule = None
    else:
        encoded = _mapping(schedule_value, "curriculum schedule")
        _exact_keys(encoded, _SCHEDULE_KEYS, "curriculum schedule")
        schedule = CurriculumSchedule(
            start_step=_plain_int(encoded["start_step"], "start_step", minimum=0),
            end_step=_plain_int(encoded["end_step"], "end_step", minimum=0),
            start_probability=_finite_real(
                encoded["start_probability"], "start_probability"
            ),
            end_probability=_finite_real(
                encoded["end_probability"], "end_probability"
            ),
        )
    mode_value = document["routing_mode"]
    if not isinstance(mode_value, str):
        raise TypeError("routing_mode must be a string")
    try:
        mode = RoutingMode(mode_value)
    except ValueError as error:
        raise ValueError("routing_mode is unsupported") from error
    surrogate = document["straight_through_route_surrogate"]
    if type(surrogate) is not bool:
        raise TypeError("straight_through_route_surrogate must be boolean")
    config = BindingModelConfig(
        task=task,
        d_model=_plain_int(
            document["d_model"],
            "d_model",
            minimum=1,
            maximum=_MAX_MODEL_DIMENSION,
        ),
        cp_rank=_plain_int(
            document["cp_rank"],
            "cp_rank",
            minimum=1,
            maximum=_MAX_MODEL_DIMENSION,
        ),
        router_hidden_dim=_plain_int(
            document["router_hidden_dim"],
            "router_hidden_dim",
            minimum=1,
            maximum=_MAX_MODEL_DIMENSION,
        ),
        routing_mode=mode,
        curriculum_schedule=schedule,
        curriculum_seed=_plain_int(
            document["curriculum_seed"],
            "curriculum_seed",
            minimum=-(1 << 63),
            maximum=(1 << 63) - 1,
        ),
        scale_feature_dim=_plain_int(
            document["scale_feature_dim"],
            "scale_feature_dim",
            minimum=1,
            maximum=_MAX_MODEL_DIMENSION,
        ),
        straight_through_route_surrogate=surrogate,
    )
    try:
        with torch.device("meta"):
            schema = RoutedBindingModel(config)
        elements = sum(tensor.numel() for tensor in schema.state_dict().values())
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("model config has invalid dimensions") from error
    if elements > _MAX_MODEL_ELEMENTS:
        raise ValueError("model config exceeds the allocation safety limit")
    return config


def _load_fixture(data: bytes) -> BindingModelInputs:
    root = _mapping(_strict_json(data, "fixture"), "fixture")
    if set(root) != {"schema_version", "inputs"} or root["schema_version"] != 1:
        raise ValueError("invalid fixture schema")
    encoded_inputs = _mapping(root["inputs"], "fixture.inputs")
    if set(encoded_inputs) != set(_INPUT_DTYPES):
        raise ValueError("invalid fixture input fields")
    tensors: dict[str, Tensor] = {}
    common_shape: tuple[int, int] | None = None
    for name, dtype in _INPUT_DTYPES.items():
        encoded = _mapping(encoded_inputs[name], f"fixture.inputs.{name}")
        if set(encoded) != {"dtype", "shape", "values"}:
            raise ValueError(f"invalid fixture tensor schema for {name}")
        dtype_name = "bool" if dtype is torch.bool else "int64"
        if encoded["dtype"] != dtype_name:
            raise ValueError(f"fixture {name} must declare dtype {dtype_name}")
        shape = encoded["shape"]
        if (
            type(shape) is not list
            or len(shape) != 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        ):
            raise ValueError(f"fixture {name} shape must contain two positive integers")
        dimensions = (shape[0], shape[1])
        if dimensions[0] > _MAX_BATCH_SIZE or dimensions[1] > _MAX_TIME_STEPS:
            raise ValueError(f"fixture {name} dimensions exceed safety limits")
        if dimensions[0] * dimensions[1] > _MAX_INPUT_ELEMENTS:
            raise ValueError(f"fixture {name} exceeds the element safety limit")
        if common_shape is None:
            common_shape = dimensions
        elif dimensions != common_shape:
            raise ValueError("all fixture tensors must share shape [N,T]")
        rows = encoded["values"]
        if type(rows) is not list or len(rows) != dimensions[0]:
            raise ValueError(f"fixture {name} values do not match its shape")
        for row in rows:
            if type(row) is not list or len(row) != dimensions[1]:
                raise ValueError(f"fixture {name} values do not match its shape")
            if dtype is torch.bool:
                if any(type(item) is not bool for item in row):
                    raise ValueError(f"fixture {name} values must be booleans")
            elif any(type(item) is not int for item in row):
                raise ValueError(f"fixture {name} values must be integers")
        tensors[name] = torch.tensor(rows, dtype=dtype)
    inputs = BindingModelInputs(**tensors)
    # The model performs vocabulary/range validation during the first warmup.
    return inputs


def _git(repository_root: Path, *arguments: str) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={repository_root.resolve().as_posix()}",
        *arguments,
    ]
    try:
        result = subprocess.run(
            command,
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("measurement worker requires an accessible Git checkout") from error
    return result.stdout.strip()


def _bind_checkout(
    repository_root: Path, expected_commit: str, expected_tree: str
) -> dict[str, object]:
    package_root = (repository_root / "v3" / "src" / "tnlm_v3").resolve()
    package_file = Path(tnlm_v3.__file__).resolve()
    if not package_file.is_relative_to(package_root):
        raise RuntimeError(f"tnlm_v3 imported outside the checkout: {package_file}")
    head = _git(repository_root, "rev-parse", "HEAD").lower()
    tree = _git(repository_root, "rev-parse", "HEAD^{tree}").lower()
    if head != expected_commit or tree != expected_tree:
        raise ValueError("checkout commit/tree does not match the trusted expectation")
    worktree_status = _git(
        repository_root, "status", "--porcelain", "--untracked-files=all"
    )
    if worktree_status:
        raise RuntimeError("measurement worker requires a completely clean worktree")
    return {
        "code_commit": head,
        "code_tree": tree,
        "package_file": str(package_file),
        "package_file_sha256": _sha256(package_file.read_bytes()),
        "worker_file": str(Path(__file__).resolve()),
        "worker_file_sha256": _sha256(Path(__file__).resolve().read_bytes()),
        "worktree_clean": True,
    }


def _load_checkpoint_model(config_data: bytes, checkpoint_data: bytes) -> RoutedBindingModel:
    config_document = _strict_json(config_data, "model config", canonical=True)
    config = _parse_model_config(config_document)
    try:
        loaded = torch.load(
            io.BytesIO(checkpoint_data), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise ValueError("checkpoint is not a valid weights-only Torch payload") from error
    state_value: object = loaded
    if isinstance(loaded, Mapping) and set(loaded) == {"state_dict"}:
        state_value = loaded["state_dict"]
    if not isinstance(state_value, Mapping) or any(
        not isinstance(name, str) or not isinstance(tensor, Tensor)
        for name, tensor in state_value.items()
    ):
        raise ValueError("checkpoint must contain only a string-to-tensor state_dict")
    model = RoutedBindingModel(config)
    try:
        model.load_state_dict(dict(state_value), strict=True)
    except RuntimeError as error:
        raise ValueError("checkpoint state does not match the declared model config") from error
    return model


def _load_model(args: argparse.Namespace) -> tuple[RoutedBindingModel, dict[str, object]]:
    if args.variant == "compact":
        artifact_data = _read_bounded(args.artifact, _MAX_MODEL_BYTES, "artifact")
        artifact_sha = _sha256(artifact_data)
        if artifact_sha != args.expected_artifact_sha256:
            raise ValueError("artifact SHA-256 does not match the trusted expectation")
        model, manifest, selection = deserialize_compact_binding_model(
            artifact_data,
            expected_source_fingerprint=args.expected_source_fingerprint,
            expected_manifest_fingerprint=args.expected_manifest_fingerprint,
            expected_selection_fingerprint=args.expected_selection_fingerprint,
            device="cpu",
        )
        provenance = {
            "artifact_path": str(args.artifact),
            "artifact_sha256": artifact_sha,
            "source_model_fingerprint": manifest.source_model_fingerprint,
            "manifest_fingerprint": manifest.fingerprint(),
            "selection_fingerprint": selection.fingerprint(),
        }
    else:
        config_data = _read_bounded(
            args.model_config_json, _MAX_MODEL_CONFIG_BYTES, "model config"
        )
        checkpoint_data = _read_bounded(
            args.checkpoint, _MAX_MODEL_BYTES, "checkpoint"
        )
        checkpoint_sha = _sha256(checkpoint_data)
        if checkpoint_sha != args.expected_checkpoint_sha256:
            raise ValueError("checkpoint SHA-256 does not match the trusted expectation")
        model = _load_checkpoint_model(config_data, checkpoint_data)
        provenance = {
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "model_config_path": str(args.model_config_json),
            "model_config_sha256": _sha256(config_data),
        }
    actual = model_state_fingerprint(model)
    if actual != args.expected_model_fingerprint:
        raise ValueError("loaded model fingerprint does not match the trusted expectation")
    model.eval()
    return model, {**provenance, "model_fingerprint": actual}


def _measure(
    model: RoutedBindingModel,
    inputs: BindingModelInputs,
    *,
    warmups: int,
    iterations: int,
    sample_period_ms: float,
) -> tuple[dict[str, object], dict[str, object]]:
    process = psutil.Process(os.getpid())
    loaded_rss = int(process.memory_info().rss)
    output = None
    with torch.inference_mode():
        for _ in range(warmups):
            output = model(inputs, implementation="streaming")
    gc.collect()
    warmed_rss = int(process.memory_info().rss)
    peak = [warmed_rss]
    sampler_error: list[str] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(sample_period_ms / 1000.0):
            try:
                peak[0] = max(peak[0], int(process.memory_info().rss))
            except psutil.Error as error:
                sampler_error.append(str(error))
                return

    sampler = threading.Thread(target=sample, name="rss-sampler", daemon=True)
    elapsed_samples: list[int] = []
    sampler.start()
    try:
        with torch.inference_mode():
            for _ in range(iterations):
                started = time.perf_counter_ns()
                output = model(inputs, implementation="streaming")
                elapsed_samples.append(time.perf_counter_ns() - started)
    finally:
        stop.set()
        sampler.join(timeout=max(1.0, sample_period_ms / 1000.0 * 4))
    if sampler.is_alive():
        raise RuntimeError("RSS sampler did not terminate")
    if sampler_error:
        raise RuntimeError(f"RSS sampling failed: {sampler_error[0]}")
    final_rss = int(process.memory_info().rss)
    # This is the peak sampled during the timed region, relative to the warmed
    # baseline.  Keep the independently observed loaded RSS out of this value:
    # a transiently larger load-time footprint is not a forward-workspace peak.
    sampled_peak = max(peak[0], final_rss, warmed_rss)
    total_ns = sum(elapsed_samples)
    if total_ns <= 0:
        raise RuntimeError("timed inference duration must be positive")
    valid_events = int(inputs.valid_mask.sum().item())
    tensor_positions = int(inputs.valid_mask.numel())
    seconds = total_ns / 1_000_000_000.0
    assert output is not None
    output_evidence = {
        "routes_sha256": _sha256(
            output.routes.detach().cpu().contiguous().numpy().tobytes()
        ),
        "value_logits_sha256": _sha256(
            output.value_logits.detach().cpu().contiguous().numpy().tobytes()
        ),
    }
    measurement = {
        "warmup_iterations": warmups,
        "timed_iterations": iterations,
        "torch_threads": torch.get_num_threads(),
        "rss_sample_period_ms": sample_period_ms,
        "loaded_rss_bytes": loaded_rss,
        "warmed_rss_bytes": warmed_rss,
        "sampled_peak_rss_bytes": sampled_peak,
        "incremental_sampled_peak_bytes": max(0, sampled_peak - warmed_rss),
        "elapsed_ns_samples": elapsed_samples,
        "valid_events_per_iteration": valid_events,
        "tensor_positions_per_iteration": tensor_positions,
        "valid_events_per_second": valid_events * iterations / seconds,
        "tensor_positions_per_second": tensor_positions * iterations / seconds,
    }
    return measurement, output_evidence


def _positive_int(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= _MAX_ITERATIONS:
        raise argparse.ArgumentTypeError(f"must lie in [1,{_MAX_ITERATIONS}]")
    return parsed


def _thread_count(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= _MAX_THREADS:
        raise argparse.ArgumentTypeError(f"must lie in [1,{_MAX_THREADS}]")
    return parsed


def _sample_period(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not _MIN_SAMPLE_PERIOD_MS <= parsed <= _MAX_SAMPLE_PERIOD_MS:
        raise argparse.ArgumentTypeError(
            f"must lie in [{_MIN_SAMPLE_PERIOD_MS},{_MAX_SAMPLE_PERIOD_MS}]"
        )
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("source", "dense_selected", "compact"), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--expected-model-fingerprint", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-code-tree", required=True)
    parser.add_argument("--warmup-iterations", type=_positive_int, required=True)
    parser.add_argument("--timed-iterations", type=_positive_int, required=True)
    parser.add_argument("--torch-threads", type=_thread_count, required=True)
    parser.add_argument("--rss-sample-period-ms", type=_sample_period, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--model-config-json", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--expected-artifact-sha256")
    parser.add_argument("--expected-source-fingerprint")
    parser.add_argument("--expected-manifest-fingerprint")
    parser.add_argument("--expected-selection-fingerprint")
    args = parser.parse_args()

    path_names = ["fixture", "output"]
    if args.variant == "compact":
        required = (
            "artifact",
            "expected_artifact_sha256",
            "expected_source_fingerprint",
            "expected_manifest_fingerprint",
            "expected_selection_fingerprint",
        )
        forbidden = ("checkpoint", "expected_checkpoint_sha256", "model_config_json")
        path_names.append("artifact")
    else:
        required = ("checkpoint", "expected_checkpoint_sha256", "model_config_json")
        forbidden = (
            "artifact",
            "expected_artifact_sha256",
            "expected_source_fingerprint",
            "expected_manifest_fingerprint",
            "expected_selection_fingerprint",
        )
        path_names.extend(("checkpoint", "model_config_json"))
    missing = [name for name in required if getattr(args, name) is None]
    unexpected = [name for name in forbidden if getattr(args, name) is not None]
    if missing or unexpected:
        parser.error(f"variant arguments invalid; missing={missing}, unexpected={unexpected}")
    for name in path_names:
        path = getattr(args, name)
        if not path.is_absolute():
            parser.error(f"--{name.replace('_', '-')} must be an absolute path")
        setattr(args, name, path.resolve())
    resolved_paths = [getattr(args, name) for name in path_names]
    if len(set(resolved_paths)) != len(resolved_paths):
        parser.error("fixture, model inputs, and output must be distinct paths")
    for index, left in enumerate(resolved_paths):
        for right in resolved_paths[index + 1 :]:
            try:
                same_file = left.exists() and right.exists() and left.samefile(right)
            except OSError as error:
                parser.error(f"cannot establish path identity: {error}")
            if same_file:
                parser.error(
                    "fixture, model inputs, and output must not be hard links "
                    "to the same file"
                )
    repository_root = Path(__file__).resolve().parents[2]
    if any(path.is_relative_to(repository_root) for path in resolved_paths):
        parser.error("fixture, model inputs, and output must be outside the source checkout")
    input_paths = {getattr(args, name) for name in path_names if name != "output"}
    if args.output in input_paths:
        parser.error("--output must be distinct from every input path")
    try:
        args.expected_fixture_sha256 = _strict_digest(
            args.expected_fixture_sha256, "--expected-fixture-sha256"
        )
        args.expected_model_fingerprint = _strict_digest(
            args.expected_model_fingerprint, "--expected-model-fingerprint"
        )
        args.expected_code_commit = _strict_digest(
            args.expected_code_commit, "--expected-code-commit", 40
        )
        args.expected_code_tree = _strict_digest(
            args.expected_code_tree, "--expected-code-tree", 40
        )
        for name in (
            "expected_checkpoint_sha256",
            "expected_artifact_sha256",
            "expected_source_fingerprint",
            "expected_manifest_fingerprint",
            "expected_selection_fingerprint",
        ):
            value = getattr(args, name)
            if value is not None:
                setattr(args, name, _strict_digest(value, "--" + name.replace("_", "-")))
    except ValueError as error:
        parser.error(str(error))
    return args


def _execute(args: argparse.Namespace) -> tuple[int, dict[str, object]]:
    record: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "status": "in_progress",
        "scope": "isolated_runtime_rss_measurement_not_a_comparative_win_gate",
        "variant": args.variant,
        "implementation": "streaming",
        "stage": "preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid(),
    }
    _atomic_json(args.output, record)
    repository_root = Path(__file__).resolve().parents[2]
    try:
        torch.set_num_threads(args.torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        checkout = _bind_checkout(
            repository_root, args.expected_code_commit, args.expected_code_tree
        )
        record.update({"stage": "fixture", "checkout": checkout})
        _atomic_json(args.output, record)
        fixture_data = _read_bounded(args.fixture, _MAX_FIXTURE_BYTES, "fixture")
        fixture_sha = _sha256(fixture_data)
        if fixture_sha != args.expected_fixture_sha256:
            raise ValueError("fixture SHA-256 does not match the trusted expectation")
        inputs = _load_fixture(fixture_data)
        record.update(
            {
                "stage": "model_load",
                "fixture": {"path": str(args.fixture), "sha256": fixture_sha},
            }
        )
        _atomic_json(args.output, record)
        model, model_provenance = _load_model(args)
        record.update({"stage": "measurement", "model": model_provenance})
        _atomic_json(args.output, record)
        measurement, output_evidence = _measure(
            model,
            inputs,
            warmups=args.warmup_iterations,
            iterations=args.timed_iterations,
            sample_period_ms=args.rss_sample_period_ms,
        )
        final_checkout = _bind_checkout(
            repository_root, args.expected_code_commit, args.expected_code_tree
        )
        if final_checkout["worker_file_sha256"] != checkout["worker_file_sha256"]:
            raise RuntimeError("measurement worker changed during execution")
        record.update(
            {
                "status": "passed",
                "stage": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "measurement": measurement,
                "output_evidence": output_evidence,
                "environment": {
                    "python": platform.python_version(),
                    "python_executable": sys.executable,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "psutil": psutil.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "torch_interop_threads": torch.get_num_interop_threads(),
                },
            }
        )
        _atomic_json(args.output, record)
        return 0, record
    except Exception as error:
        record.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _atomic_json(args.output, record)
        return 1, record


def main() -> None:
    args = _parse_args()
    code, record = _execute(args)
    sys.stdout.buffer.write(_encoded_json(record, pretty=False))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
