"""Dependency-free structural types for aggregated binary sampling."""

from typing import (
    Any,
    Mapping,
    NotRequired,
    Protocol,
    TypedDict,
    TypeVar,
)


SamplerConfig = Mapping[str, Any]


class SamplerIdentity(TypedDict):
    """Stable implementation identity embedded in a sample set."""

    name: str
    version: str
    backend: NotRequired[str]


class BinarySampleRecord(TypedDict):
    """One distinct binary sample and its positive aggregate multiplicity."""

    sample: list[int]
    occurrences: int
    metadata: NotRequired[dict[str, Any]]


class BinarySampleSet(TypedDict):
    """Static shape of ``binary-sample-set.v1``."""

    schema: str
    problem_id: str
    sample_representation: str
    variable_count: int
    distribution_kind: str
    shots: int | None
    records: list[BinarySampleRecord]
    sampler: SamplerIdentity
    seed: int | None
    runtime_seconds: int | float
    metadata: NotRequired[dict[str, Any]]


ProblemT_contra = TypeVar('ProblemT_contra', contravariant=True)
SampleSetT_co = TypeVar('SampleSetT_co', covariant=True)


class Sampler(Protocol[ProblemT_contra, SampleSetT_co]):
    """Representation-neutral structural interface for sampling algorithms.

    Implementations need not inherit from this protocol.  A solver may expose
    both ``sample()`` for its full aggregated distribution and ``solve()`` for
    a best-candidate result without changing either wire contract.
    """

    def sample(
        self,
        problem: ProblemT_contra,
        config: SamplerConfig | None = None,
    ) -> SampleSetT_co:
        """Sample without mutating the problem or configuration."""


__all__ = [
    'BinarySampleRecord',
    'BinarySampleSet',
    'Sampler',
    'SamplerConfig',
    'SamplerIdentity',
]
