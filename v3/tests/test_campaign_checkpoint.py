from __future__ import annotations

import hashlib
import json
import random
import struct
from typing import Any, Callable

import pytest
import torch
from torch import nn
import torch.optim.optimizer as optimizer_hooks

from tnlm_v3.baselines import (
    CachedCausalTransformerBindingBaseline,
    CachedTransformerBindingBaselineConfig,
    RecurrentBindingBaseline,
    RecurrentBindingBaselineConfig,
)
from tnlm_v3.binding import BindingArchitectureConfig, BindingModelConfig, RoutedBindingModel
from tnlm_v3.campaign import train_baseline_step
from tnlm_v3.campaign_checkpoint import (
    CampaignCheckpointContract,
    CampaignResumeState,
    campaign_checkpoint_contract,
    campaign_model_fingerprint,
    deserialize_campaign_checkpoint,
    serialize_campaign_checkpoint,
)
from tnlm_v3.causal_ttn import CausalCompleteTreeBindingBaseline, CausalTreeBindingBaselineConfig
from tnlm_v3.data import BindingTaskConfig, collate_binding_episodes, generate_binding_episodes
from tnlm_v3.routing import CurriculumSchedule, RoutingMode
from tnlm_v3.training import train_binding_step


PREFIX = struct.Struct("<8sIQQ32s32s")
RUN_SHA = "a" * 64
STREAM_SHA = "b" * 64
MODEL_KINDS = ("routed", "gru", "cached_transformer", "causal_ttn")


def task() -> BindingTaskConfig:
    return BindingTaskConfig(
        num_surface_keys=5,
        value_cardinality=4,
        branches=3,
        max_live_bindings=3,
        min_length=10,
        max_length=16,
        heldout_key_value_pairs=((0, 0),),
        global_distractor_probability=0.5,
    )


def model(kind: str) -> nn.Module:
    architecture = BindingArchitectureConfig.from_task(task())
    torch.manual_seed(7_931)
    if kind == "routed":
        return RoutedBindingModel(
            BindingModelConfig(
                architecture,
                d_model=8,
                cp_rank=4,
                router_hidden_dim=8,
                routing_mode=RoutingMode.CURRICULUM,
                curriculum_schedule=CurriculumSchedule(0, 8, 1.0, 0.0),
                curriculum_seed=17,
                scale_feature_dim=4,
            )
        ).double()
    if kind == "gru":
        return RecurrentBindingBaseline(
            RecurrentBindingBaselineConfig(
                architecture, d_model=8, hidden_dim=12, num_layers=2
            )
        ).double()
    if kind == "cached_transformer":
        return CachedCausalTransformerBindingBaseline(
            CachedTransformerBindingBaselineConfig(
                architecture,
                d_model=8,
                num_heads=2,
                num_layers=2,
                ff_dim=16,
            )
        ).double()
    if kind == "causal_ttn":
        return CausalCompleteTreeBindingBaseline(
            CausalTreeBindingBaselineConfig(
                architecture, d_model=8, cp_rank=4, scale_feature_dim=4
            )
        ).double()
    raise AssertionError(kind)


def batch(seed: int):
    return collate_binding_episodes(
        generate_binding_episodes(
            task(), count=2, seed=seed, split="train", lengths=(16, 12)
        )
    )


def step(model: nn.Module, optimizer: torch.optim.Optimizer, seed: int, index: int):
    value = batch(seed)
    if type(model) is RoutedBindingModel:
        return train_binding_step(model, value, optimizer, training_step=index)
    return train_baseline_step(model, value, optimizer)


def resume(index: int = 0) -> CampaignResumeState:
    return CampaignResumeState(index, index, RUN_SHA, STREAM_SHA)


def contract(model: nn.Module, optimizer: torch.optim.Optimizer) -> CampaignCheckpointContract:
    fresh = type(model)(model.config).to(
        device="cpu", dtype=next(model.parameters()).dtype
    )
    fresh_optimizer = torch.optim.AdamW(
        fresh.parameters(),
        lr=optimizer.param_groups[0]["lr"],
        weight_decay=optimizer.param_groups[0]["weight_decay"],
    )
    return campaign_checkpoint_contract(fresh, fresh_optimizer)


def restore(
    blob: bytes | bytearray | memoryview,
    expected_contract: CampaignCheckpointContract,
    **kwargs: object,
):
    return deserialize_campaign_checkpoint(
        blob,
        expected_run_spec_sha256=RUN_SHA,
        expected_stream_prefix_sha256=STREAM_SHA,
        expected_contract=expected_contract,
        **kwargs,
    )


def nested_equal(left: Any, right: Any) -> bool:
    if type(left) is torch.Tensor:
        return type(right) is torch.Tensor and torch.equal(left, right)
    if type(left) is dict:
        return type(right) is dict and left.keys() == right.keys() and all(
            nested_equal(left[key], right[key]) for key in left
        )
    if type(left) in (list, tuple):
        return type(left) is type(right) and len(left) == len(right) and all(
            nested_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def split_blob(blob: bytes) -> tuple[tuple[Any, ...], bytes, bytes]:
    prefix = PREFIX.unpack_from(blob)
    header_size, payload_size = prefix[2], prefix[3]
    header = blob[PREFIX.size : PREFIX.size + header_size]
    payload = blob[PREFIX.size + header_size :]
    assert len(payload) == payload_size
    return prefix, header, payload


def repack(header: bytes, payload: bytes, *, prefix: tuple[Any, ...] | None = None) -> bytes:
    magic, version = (b"TNLM4CK\x00", 1) if prefix is None else prefix[:2]
    return PREFIX.pack(
        magic,
        version,
        len(header),
        len(payload),
        hashlib.sha256(header).digest(),
        hashlib.sha256(payload).digest(),
    ) + header + payload


def changed_header(blob: bytes, mutate: Callable[[dict[str, Any]], None]) -> bytes:
    prefix, header, payload = split_blob(blob)
    value = json.loads(header)
    mutate(value)
    encoded = json.dumps(
        value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return repack(encoded, payload, prefix=prefix)


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_uninitialized_round_trip_is_canonical_and_preserves_exact_model(kind: str) -> None:
    original = model(kind)
    original.train(False)
    optimizer = torch.optim.AdamW(original.parameters(), lr=7e-4, weight_decay=0.03)
    before_torch = torch.random.get_rng_state().clone()
    before_python = random.getstate()

    first = serialize_campaign_checkpoint(original, optimizer, resume())
    second = serialize_campaign_checkpoint(original, optimizer, resume())

    assert first == second
    assert torch.equal(torch.random.get_rng_state(), before_torch)
    assert random.getstate() == before_python
    restored, restored_optimizer, restored_resume = restore(
        first, contract(original, optimizer), device="cpu"
    )
    assert type(restored) is type(original)
    assert restored_resume == resume()
    assert campaign_model_fingerprint(restored) == campaign_model_fingerprint(original)
    assert all(not module.training for module in restored.modules())
    assert nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_initialized_checkpoint_replays_next_step_bit_exactly(kind: str) -> None:
    original = model(kind)
    optimizer = torch.optim.AdamW(original.parameters(), lr=7e-4, weight_decay=0.03)
    step(original, optimizer, 101, 0)
    optimizer.zero_grad(set_to_none=True)
    blob = serialize_campaign_checkpoint(original, optimizer, resume(1))
    restored, restored_optimizer, _ = restore(blob, contract(original, optimizer))
    assert serialize_campaign_checkpoint(restored, restored_optimizer, resume(1)) == blob

    original_output, original_loss = step(original, optimizer, 102, 1)
    restored_output, restored_loss = step(restored, restored_optimizer, 102, 1)

    assert torch.equal(original_output.value_logits, restored_output.value_logits)
    assert torch.equal(original_loss.total, restored_loss.total)
    assert all(
        torch.equal(left, right)
        for left, right in zip(original.parameters(), restored.parameters(), strict=True)
    )
    assert nested_equal(optimizer.state_dict(), restored_optimizer.state_dict())


def test_resume_state_is_strict_and_self_consistent() -> None:
    with pytest.raises(ValueError, match="must match"):
        CampaignResumeState(2, 1, RUN_SHA, STREAM_SHA)
    with pytest.raises(ValueError, match="present together"):
        CampaignResumeState(1, 1, RUN_SHA, STREAM_SHA, best_metric=0.5)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        CampaignResumeState(1, 1, RUN_SHA, STREAM_SHA, 1.1, 1)
    with pytest.raises((TypeError, ValueError), match="best_step"):
        CampaignResumeState(1, 1, RUN_SHA, STREAM_SHA, 0.5, 2)
    with pytest.raises(ValueError, match="at most"):
        CampaignResumeState(1 << 24, 1 << 24, RUN_SHA, STREAM_SHA)
    assert CampaignResumeState(3, 3, RUN_SHA, STREAM_SHA, 0.75, 2).best_step == 2


def test_success_restores_python_and_torch_rng_and_failure_rolls_back() -> None:
    original = model("gru")
    optimizer = torch.optim.AdamW(original.parameters())
    torch.manual_seed(91_001)
    random.seed(91_002)
    random.gauss(0.0, 1.0)
    expected_torch = torch.random.get_rng_state().clone()
    expected_python = random.getstate()
    blob = serialize_campaign_checkpoint(original, optimizer, resume())

    torch.manual_seed(88)
    random.seed(89)
    expected = contract(original, optimizer)
    restore(blob, expected)
    assert torch.equal(torch.random.get_rng_state(), expected_torch)
    assert random.getstate() == expected_python

    torch.manual_seed(177)
    random.seed(178)
    caller_torch = torch.random.get_rng_state().clone()
    caller_python = random.getstate()
    forged = changed_header(
        blob, lambda header: header.__setitem__("model_fingerprint", "0" * 64)
    )
    with pytest.raises(ValueError, match="fingerprint"):
        restore(forged, expected)
    assert torch.equal(torch.random.get_rng_state(), caller_torch)
    assert random.getstate() == caller_python


def test_expected_provenance_is_fail_closed() -> None:
    original = model("gru")
    optimizer = torch.optim.AdamW(original.parameters())
    expected = contract(original, optimizer)
    blob = serialize_campaign_checkpoint(original, optimizer, resume())
    with pytest.raises(ValueError, match="run specification"):
        deserialize_campaign_checkpoint(
            blob,
            expected_run_spec_sha256="c" * 64,
            expected_stream_prefix_sha256=STREAM_SHA,
            expected_contract=expected,
        )
    with pytest.raises(ValueError, match="stream prefix"):
        deserialize_campaign_checkpoint(
            blob,
            expected_run_spec_sha256=RUN_SHA,
            expected_stream_prefix_sha256="d" * 64,
            expected_contract=expected,
        )
    with pytest.raises(ValueError, match="only CPU"):
        restore(blob, expected, device="meta")

    other = model("causal_ttn")
    other_optimizer = torch.optim.AdamW(other.parameters())
    with pytest.raises(ValueError, match="executable contract"):
        restore(blob, contract(other, other_optimizer))


def test_contract_binds_the_complete_optimizer_execution_settings() -> None:
    original = model("gru")
    trusted_optimizer = torch.optim.AdamW(
        original.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,
        fused=False,
    )
    trusted = contract(original, trusted_optimizer)
    attacker_optimizer = torch.optim.AdamW(
        original.parameters(),
        lr=1e-3,
        betas=(0.1, 0.2),
        eps=0.25,
        weight_decay=0.01,
        foreach=None,
        fused=None,
    )
    blob = serialize_campaign_checkpoint(original, attacker_optimizer, resume())
    with pytest.raises(ValueError, match="executable contract"):
        restore(blob, trusted)


def test_loader_never_calls_pytorch_archive_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    original = model("gru")
    optimizer = torch.optim.AdamW(original.parameters())
    expected = contract(original, optimizer)
    blob = serialize_campaign_checkpoint(original, optimizer, resume())

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("executable PyTorch archive API was called")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    restored, _, _ = restore(blob, expected)
    assert campaign_model_fingerprint(restored) == campaign_model_fingerprint(original)


def test_bytes_subclasses_cannot_execute_or_bypass_the_input_cap() -> None:
    original = model("gru")
    optimizer = torch.optim.AdamW(original.parameters())
    expected = contract(original, optimizer)
    blob = serialize_campaign_checkpoint(original, optimizer, resume())
    calls: list[int] = []

    class Evil(bytearray):
        def __bytes__(self) -> bytes:
            calls.append(1)
            return b"x" * (70 * 1024 * 1024)

    hostile = Evil(blob)
    with pytest.raises(TypeError, match="bytes-like"):
        restore(hostile, expected)
    assert calls == []
    restored, _, _ = restore(memoryview(hostile), expected)
    assert calls == []
    assert campaign_model_fingerprint(restored) == campaign_model_fingerprint(original)


def test_restore_forces_cpu_independent_of_ambient_default_device() -> None:
    original = model("gru")
    optimizer = torch.optim.AdamW(original.parameters())
    expected = contract(original, optimizer)
    blob = serialize_campaign_checkpoint(original, optimizer, resume())
    prior = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        restored, restored_optimizer, _ = restore(blob, expected, device="cpu")
    finally:
        torch.set_default_device(prior)
    assert {parameter.device.type for parameter in restored.parameters()} == {"cpu"}
    assert all(
        tensor.device.type == "cpu"
        for state in restored_optimizer.state.values()
        for tensor in state.values()
    )


@pytest.mark.parametrize("mutation", ["truncated", "trailing", "payload", "magic", "version"])
def test_envelope_corruption_is_rejected(mutation: str) -> None:
    original = model("gru")
    optimizer = torch.optim.AdamW(original.parameters())
    expected = contract(original, optimizer)
    blob = serialize_campaign_checkpoint(original, optimizer, resume())
    if mutation == "truncated":
        changed = blob[:-1]
    elif mutation == "trailing":
        changed = blob + b"x"
    elif mutation == "payload":
        changed = blob[:-1] + bytes([blob[-1] ^ 1])
    elif mutation == "magic":
        changed = b"XXXXXXXX" + blob[8:]
    else:
        changed = blob[:8] + struct.pack("<I", 2) + blob[12:]
    with pytest.raises(ValueError):
        restore(changed, expected)


def test_duplicate_noncanonical_deep_and_unknown_json_are_rejected() -> None:
    original = model("gru")
    optimizer = torch.optim.AdamW(original.parameters())
    expected = contract(original, optimizer)
    blob = serialize_campaign_checkpoint(original, optimizer, resume())
    prefix, header, payload = split_blob(blob)
    duplicate = header.replace(
        b'"artifact_kind":', b'"artifact_kind":"wrong","artifact_kind":', 1
    )
    for changed, message in (
        (repack(duplicate, payload, prefix=prefix), "duplicate"),
        (repack(b" " + header, payload, prefix=prefix), "canonical"),
        (changed_header(blob, lambda value: value.__setitem__("extra", 1)), "fields"),
    ):
        with pytest.raises(ValueError, match=message):
            restore(changed, expected)
    deep = json.loads(header)
    nested: object = 0
    for _ in range(40):
        nested = [nested]
    deep["extra"] = nested
    encoded = json.dumps(deep, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="deeply"):
        restore(repack(encoded, payload, prefix=prefix), expected)


def test_late_tensor_schema_forgery_is_rejected_before_model_use() -> None:
    original = model("gru")
    optimizer = torch.optim.AdamW(original.parameters())
    expected = contract(original, optimizer)
    blob = serialize_campaign_checkpoint(original, optimizer, resume())

    def mutate(header: dict[str, Any]) -> None:
        record = header["tensors"][0]
        record["shape"] = [268_435_457]

    with pytest.raises(ValueError, match="large|byte count"):
        restore(changed_header(blob, mutate), expected)


def test_exact_model_type_metadata_and_instance_callables_are_enforced() -> None:
    base = model("gru")

    class Derived(RecurrentBindingBaseline):
        pass

    derived = Derived(base.config).double()
    with pytest.raises(TypeError, match="exact"):
        campaign_model_fingerprint(derived)

    mutated = model("gru")
    mutated.encoder.norm.eps = 0.5
    with pytest.raises(ValueError, match="metadata"):
        campaign_model_fingerprint(mutated)

    callable_model = model("gru")
    callable_model.forward = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="callable"):
        campaign_model_fingerprint(callable_model)


def test_local_module_parameter_optimizer_and_global_hooks_are_rejected() -> None:
    original = model("gru")
    optimizer = torch.optim.AdamW(original.parameters())
    module_handle = original.encoder.norm.register_forward_hook(
        lambda _module, _inputs, output: output
    )
    try:
        with pytest.raises(ValueError, match="runtime hooks"):
            serialize_campaign_checkpoint(original, optimizer, resume())
    finally:
        module_handle.remove()

    state_dict_calls: list[int] = []
    state_dict_handle = original.register_state_dict_pre_hook(
        lambda *_args: state_dict_calls.append(1)
    )
    try:
        with pytest.raises(ValueError, match="runtime hooks"):
            serialize_campaign_checkpoint(original, optimizer, resume())
        assert state_dict_calls == []
    finally:
        state_dict_handle.remove()

    parameter = next(original.parameters())
    parameter_handle = parameter.register_hook(lambda gradient: gradient)
    try:
        with pytest.raises(ValueError, match="parameter hooks"):
            serialize_campaign_checkpoint(original, optimizer, resume())
    finally:
        parameter_handle.remove()

    optimizer_handle = optimizer.register_step_pre_hook(lambda *_args, **_kwargs: None)
    try:
        with pytest.raises(ValueError, match="optimizer hooks"):
            serialize_campaign_checkpoint(original, optimizer, resume())
    finally:
        optimizer_handle.remove()

    optimizer.step = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="optimizer callables"):
        serialize_campaign_checkpoint(original, optimizer, resume())
    del optimizer.step

    blob = serialize_campaign_checkpoint(original, optimizer, resume())
    expected = contract(original, optimizer)
    calls: list[int] = []
    global_handle = torch.nn.modules.module.register_module_module_registration_hook(
        lambda *_args: calls.append(1)
    )
    try:
        with pytest.raises(ValueError, match="global PyTorch module hooks"):
            restore(blob, expected)
        assert calls == []
    finally:
        global_handle.remove()

    optimizer_global_handle = optimizer_hooks.register_optimizer_step_pre_hook(
        lambda *_args, **_kwargs: None
    )
    try:
        with pytest.raises(ValueError, match="global PyTorch optimizer hooks"):
            restore(blob, expected)
    finally:
        optimizer_global_handle.remove()


def test_pending_gradients_nonfinite_values_and_wrong_optimizer_are_rejected() -> None:
    original = model("gru")
    optimizer = torch.optim.AdamW(original.parameters())
    next(original.parameters()).grad = torch.zeros_like(next(original.parameters()))
    with pytest.raises(ValueError, match="pending gradients"):
        serialize_campaign_checkpoint(original, optimizer, resume())
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        next(original.parameters()).reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        serialize_campaign_checkpoint(original, optimizer, resume())
    clean = model("gru")
    with pytest.raises(TypeError, match="AdamW"):
        serialize_campaign_checkpoint(
            clean, torch.optim.SGD(clean.parameters(), lr=0.1), resume()
        )


def test_tied_and_oversized_parameter_storage_are_rejected() -> None:
    tied = model("gru")
    assert tied.encoder.primary.weight.shape == tied.encoder.secondary.weight.shape
    tied.encoder.secondary.weight = nn.Parameter(tied.encoder.primary.weight.data)
    with pytest.raises(ValueError, match="shared tensor storage"):
        serialize_campaign_checkpoint(tied, torch.optim.AdamW(tied.parameters()), resume())

    viewed = model("gru")
    parameter = viewed.encoder.primary.weight
    base = torch.empty(parameter.numel() + 1, dtype=parameter.dtype)
    parameter.data = base[: parameter.numel()].view_as(parameter)
    with pytest.raises(ValueError, match="exact-sized"):
        serialize_campaign_checkpoint(
            viewed, torch.optim.AdamW(viewed.parameters()), resume()
        )


def test_optimizer_binding_state_schema_and_aliases_are_rejected() -> None:
    original = model("gru")
    reversed_optimizer = torch.optim.AdamW(list(reversed(list(original.parameters()))))
    with pytest.raises(ValueError, match="parameter order"):
        serialize_campaign_checkpoint(original, reversed_optimizer, resume())

    optimizer = torch.optim.AdamW(original.parameters())
    step(original, optimizer, 71, 0)
    optimizer.zero_grad(set_to_none=True)
    states = list(optimizer.state.values())
    assert len(states) >= 2
    states[1]["step"] = states[0]["step"]
    with pytest.raises(ValueError, match="shared tensor storage"):
        serialize_campaign_checkpoint(original, optimizer, resume(1))


def test_optimizer_step_and_resume_position_are_cryptographically_checked() -> None:
    original = model("gru")
    optimizer = torch.optim.AdamW(original.parameters())
    step(original, optimizer, 81, 0)
    optimizer.zero_grad(set_to_none=True)
    blob = serialize_campaign_checkpoint(original, optimizer, resume(1))
    expected = contract(original, optimizer)

    def mutate(header: dict[str, Any]) -> None:
        header["resume_state"]["global_step"] = 2
        header["resume_state"]["data_cursor"] = 2

    with pytest.raises(ValueError, match="optimizer step"):
        restore(changed_header(blob, mutate), expected)

    with pytest.raises(ValueError, match="executable contract"):
        restore(
            changed_header(
                blob,
                lambda header: header["optimizer"].__setitem__("foreach", 0),
            ),
            expected,
        )


def test_fingerprint_changes_with_config_or_parameter_bits() -> None:
    first = model("gru")
    second = model("gru")
    assert campaign_model_fingerprint(first) == campaign_model_fingerprint(second)
    with torch.no_grad():
        next(second.parameters()).reshape(-1)[0].add_(1.0)
    assert campaign_model_fingerprint(first) != campaign_model_fingerprint(second)


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_float32_round_trip_preserves_family_dtype_and_bits(kind: str) -> None:
    original = model(kind).float()
    optimizer = torch.optim.AdamW(original.parameters(), lr=1e-3)
    blob = serialize_campaign_checkpoint(original, optimizer, resume())
    restored, _, _ = restore(blob, contract(original, optimizer))
    assert type(restored) is type(original)
    assert {parameter.dtype for parameter in restored.parameters()} == {torch.float32}
    assert campaign_model_fingerprint(restored) == campaign_model_fingerprint(original)


@pytest.mark.parametrize("mode", (RoutingMode.ORACLE, RoutingMode.LATENT))
def test_other_routed_modes_preserve_exact_executable_config(mode: RoutingMode) -> None:
    architecture = BindingArchitectureConfig.from_task(task())
    original = RoutedBindingModel(
        BindingModelConfig(
            architecture,
            d_model=8,
            cp_rank=4,
            router_hidden_dim=8,
            routing_mode=mode,
            scale_feature_dim=4,
        )
    ).double()
    optimizer = torch.optim.AdamW(original.parameters())
    blob = serialize_campaign_checkpoint(original, optimizer, resume())
    restored, _, _ = restore(blob, contract(original, optimizer))
    assert type(restored) is RoutedBindingModel
    assert restored.config == original.config
    assert campaign_model_fingerprint(restored) == campaign_model_fingerprint(original)
