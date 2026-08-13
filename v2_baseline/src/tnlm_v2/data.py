from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class TaskSpec:
    name: str
    vocab: Mapping[str, int]
    num_classes: int
    max_branches: int
    description: str
    pad_token: str = "<PAD>"

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return int(self.vocab[self.pad_token])

    @property
    def inverse_vocab(self) -> Dict[int, str]:
        return {int(v): k for k, v in self.vocab.items()}


@dataclass
class TaskBatch:
    tokens: torch.LongTensor
    valid_mask: torch.BoolTensor
    routes: torch.LongTensor
    labels: torch.LongTensor
    metadata: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "TaskBatch":
        return TaskBatch(
            self.tokens.to(device),
            self.valid_mask.to(device),
            self.routes.to(device),
            self.labels.to(device),
            {k: v.to(device) for k, v in self.metadata.items()},
        )

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.tokens.shape[1])

    def slice(self, indices) -> "TaskBatch":
        return TaskBatch(
            self.tokens[indices],
            self.valid_mask[indices],
            self.routes[indices],
            self.labels[indices],
            {k: v[indices] for k, v in self.metadata.items()},
        )


class _VocabularyBuilder:
    def __init__(self) -> None:
        self.tokens: Dict[str, int] = {}

    def add(self, *names: str) -> None:
        for name in names:
            if name not in self.tokens:
                self.tokens[name] = len(self.tokens)

    def add_range(self, prefix: str, count: int) -> None:
        self.add(*(f"{prefix}{i}" for i in range(count)))

    def build(self) -> Dict[str, int]:
        return dict(self.tokens)


class BaseSyntheticTask:
    spec: TaskSpec

    @property
    def minimum_length(self) -> int:
        raise NotImplementedError

    def _generate_one(
        self, rng: np.random.Generator, sequence_length: int, active_branches: int
    ) -> Tuple[List[int], List[int], int]:
        raise NotImplementedError

    def generate(
        self,
        num_samples: int,
        sequence_length: int,
        seed: int,
        active_branches: Optional[int] = None,
    ) -> TaskBatch:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if sequence_length < self.minimum_length:
            raise ValueError(
                f"{self.spec.name} requires length >= {self.minimum_length}; got {sequence_length}"
            )
        branches = int(active_branches or self.spec.max_branches)
        if not 1 <= branches <= self.spec.max_branches:
            raise ValueError("active_branches outside task range")
        rng = np.random.default_rng(seed)
        tokens = np.full((num_samples, sequence_length), self.spec.pad_id, np.int64)
        routes = np.full((num_samples, sequence_length), -1, np.int64)
        labels = np.empty(num_samples, np.int64)
        active = np.full(num_samples, branches, np.int64)
        for row in range(num_samples):
            sample_tokens, sample_routes, label = self._generate_one(
                rng, sequence_length, branches
            )
            if len(sample_tokens) > sequence_length:
                raise RuntimeError("generator overflow")
            tokens[row, : len(sample_tokens)] = sample_tokens
            routes[row, : len(sample_routes)] = sample_routes
            labels[row] = label
        return TaskBatch(
            torch.from_numpy(tokens),
            torch.from_numpy(tokens != self.spec.pad_id),
            torch.from_numpy(routes),
            torch.from_numpy(labels),
            {"active_branches": torch.from_numpy(active)},
        )

    def decode(self, token_ids: Iterable[int]) -> List[str]:
        inv = self.spec.inverse_vocab
        return [inv[int(i)] for i in token_ids]


class InterleavedThreadsTask(BaseSyntheticTask):
    """Interleaved per-entity last-write retrieval."""

    def __init__(self, max_branches: int = 8, value_cardinality: int = 4) -> None:
        vb = _VocabularyBuilder()
        vb.add("<PAD>", "<QUERY>")
        vb.add_range("ID_", max_branches)
        vb.add_range("SET_", value_cardinality)
        self.value_cardinality = value_cardinality
        self.spec = TaskSpec(
            "interleaved_threads",
            vb.build(),
            value_cardinality,
            max_branches,
            "Randomly interleaved entity updates followed by a query for one live thread.",
        )

    @property
    def minimum_length(self) -> int:
        return 6

    def _generate_one(self, rng, sequence_length, active_branches):
        vocab = self.spec.vocab
        n_events = max(active_branches, (sequence_length - 2) // 2)
        states = np.zeros(active_branches, np.int64)
        branches = list(range(active_branches))
        branches += [
            int(x)
            for x in rng.integers(0, active_branches, size=n_events - active_branches)
        ]
        rng.shuffle(branches)
        toks: List[int] = []
        routes: List[int] = []
        for branch in branches:
            value = int(rng.integers(0, self.value_cardinality))
            states[branch] = value
            toks += [vocab[f"ID_{branch}"], vocab[f"SET_{value}"]]
            routes += [branch, branch]
        query = int(rng.integers(0, active_branches))
        toks += [vocab["<QUERY>"], vocab[f"ID_{query}"]]
        routes += [query, query]
        return toks, routes, int(states[query])


class PermutedHierarchyTask(BaseSyntheticTask):
    """Branch values are serialized in random order but reduced in a fixed tree."""

    def __init__(self, max_branches: int = 8, value_cardinality: int = 2) -> None:
        if max_branches & (max_branches - 1):
            raise ValueError("max_branches must be a power of two")
        vb = _VocabularyBuilder()
        vb.add("<PAD>", "<ROOT_QUERY>")
        vb.add_range("ID_", max_branches)
        vb.add_range("VAL_", value_cardinality)
        self.value_cardinality = value_cardinality
        self.spec = TaskSpec(
            "permuted_hierarchy",
            vb.build(),
            value_cardinality,
            max_branches,
            "Permuted branch leaves; first-scale ANDs and higher-scale parity require the correct branch tree.",
        )

    @property
    def minimum_length(self) -> int:
        return 2 * self.spec.max_branches + 1

    @staticmethod
    def _combine(left: int, right: int, level: int) -> int:
        if level == 0:
            return int(bool(left) and bool(right))
        return int(bool(left) ^ bool(right))

    def reduce_branch_values(self, values: Sequence[int]) -> int:
        current = [int(v) for v in values]
        level = 0
        while len(current) > 1:
            current = [
                self._combine(current[i], current[i + 1], level)
                for i in range(0, len(current), 2)
            ]
            level += 1
        return current[0]

    def _generate_one(self, rng, sequence_length, active_branches):
        if active_branches & (active_branches - 1):
            raise ValueError("active_branches must be a power of two")
        vocab = self.spec.vocab
        states = rng.integers(0, self.value_cardinality, size=active_branches)
        events = [(i, int(states[i])) for i in range(active_branches)]
        rng.shuffle(events)
        toks: List[int] = []
        routes: List[int] = []
        for branch, value in events:
            toks += [vocab[f"ID_{branch}"], vocab[f"VAL_{value}"]]
            routes += [branch, branch]
        toks.append(vocab["<ROOT_QUERY>"])
        routes.append(-1)
        return toks, routes, self.reduce_branch_values(states.tolist())


class PredictiveDetailTask(BaseSyntheticTask):
    """Shifted local pair contains signal XOR nuisance; nuisance is discardable."""

    def __init__(self, max_branches: int = 8, value_cardinality: int = 2) -> None:
        vb = _VocabularyBuilder()
        vb.add("<PAD>", "<QUERY>")
        vb.add_range("ID_", max_branches)
        vb.add_range("MIXA_", value_cardinality)
        vb.add_range("MIXB_", value_cardinality)
        self.value_cardinality = value_cardinality
        self.spec = TaskSpec(
            "predictive_detail",
            vb.build(),
            value_cardinality,
            max_branches,
            "ID, mixed signal, nuisance triples deliberately cross the first positional tree boundary.",
        )

    @property
    def minimum_length(self) -> int:
        return 3 * self.spec.max_branches + 2

    def _generate_one(self, rng, sequence_length, active_branches):
        vocab = self.spec.vocab
        n_events = max(active_branches, (sequence_length - 2) // 3)
        states = np.zeros(active_branches, np.int64)
        branches = list(range(active_branches))
        branches += [
            int(x)
            for x in rng.integers(0, active_branches, size=n_events - active_branches)
        ]
        rng.shuffle(branches)
        toks: List[int] = []
        routes: List[int] = []
        for branch in branches:
            signal = int(rng.integers(0, self.value_cardinality))
            nuisance = int(rng.integers(0, self.value_cardinality))
            mixed = (signal + nuisance) % self.value_cardinality
            states[branch] = signal
            toks += [
                vocab[f"ID_{branch}"],
                vocab[f"MIXA_{mixed}"],
                vocab[f"MIXB_{nuisance}"],
            ]
            routes += [branch, branch, branch]
        query = int(rng.integers(0, active_branches))
        toks += [vocab["<QUERY>"], vocab[f"ID_{query}"]]
        routes += [query, query]
        return toks, routes, int(states[query])


class CombinedLanguageTask(BaseSyntheticTask):
    """Two mixed updates per branch followed by the hierarchical root query."""

    def __init__(self, max_branches: int = 8, value_cardinality: int = 2) -> None:
        if max_branches & (max_branches - 1):
            raise ValueError("max_branches must be a power of two")
        vb = _VocabularyBuilder()
        vb.add("<PAD>", "<ROOT_QUERY>")
        vb.add_range("ID_", max_branches)
        vb.add_range("MIXSET_", value_cardinality)
        vb.add_range("MIXADD_", value_cardinality)
        vb.add_range("MIXB_", value_cardinality)
        self.value_cardinality = value_cardinality
        self.spec = TaskSpec(
            "combined_language",
            vb.build(),
            value_cardinality,
            max_branches,
            "Interleaved routed updates, local nuisance disentangling and a branch-tree root query.",
        )

    @property
    def minimum_length(self) -> int:
        return 6 * self.spec.max_branches + 1

    @staticmethod
    def _combine(left: int, right: int, level: int) -> int:
        if level == 0:
            return int(bool(left) and bool(right))
        return int(bool(left) ^ bool(right))

    def reduce_branch_values(self, values: Sequence[int]) -> int:
        current = [int(v) for v in values]
        level = 0
        while len(current) > 1:
            current = [
                self._combine(current[i], current[i + 1], level)
                for i in range(0, len(current), 2)
            ]
            level += 1
        return current[0]

    def _generate_one(self, rng, sequence_length, active_branches):
        if active_branches & (active_branches - 1):
            raise ValueError("active_branches must be a power of two")
        vocab = self.spec.vocab
        states = np.zeros(active_branches, np.int64)
        initialized = np.zeros(active_branches, bool)
        branches = list(range(active_branches)) * 2
        rng.shuffle(branches)
        toks: List[int] = []
        routes: List[int] = []
        for branch in branches:
            use_add = bool(initialized[branch])
            signal = int(rng.integers(0, self.value_cardinality))
            nuisance = int(rng.integers(0, self.value_cardinality))
            mixed = (signal + nuisance) % self.value_cardinality
            if use_add:
                states[branch] = (states[branch] + signal) % self.value_cardinality
                opname = "MIXADD"
            else:
                states[branch] = signal
                initialized[branch] = True
                opname = "MIXSET"
            toks += [
                vocab[f"ID_{branch}"],
                vocab[f"{opname}_{mixed}"],
                vocab[f"MIXB_{nuisance}"],
            ]
            routes += [branch, branch, branch]
        toks.append(vocab["<ROOT_QUERY>"])
        routes.append(-1)
        return toks, routes, self.reduce_branch_values(states.tolist())


def build_task(name: str, max_branches: int = 8) -> BaseSyntheticTask:
    name = name.strip().lower()
    factories = {
        "interleaved_threads": InterleavedThreadsTask,
        "permuted_hierarchy": PermutedHierarchyTask,
        "predictive_detail": PredictiveDetailTask,
        "combined_language": CombinedLanguageTask,
    }
    if name not in factories:
        raise KeyError(f"unknown task {name!r}")
    return factories[name](max_branches=max_branches)


def available_tasks() -> Tuple[str, ...]:
    return (
        "interleaved_threads",
        "permuted_hierarchy",
        "predictive_detail",
        "combined_language",
    )
