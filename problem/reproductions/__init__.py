"""Paper-to-solver reproduction problem definitions.

Each module owns the deterministic mathematical instances for one published
algorithm.  Solver logic remains under :mod:`lib.solvers`; notebooks only
compose these two script layers and report validation results.
"""

from .qrbnbr import (
    build_qrbnbr_s1_instance,
    build_qrbnbr_s1_suite,
)
from .scmf_qaoa import (
    build_scmf_gaussian_sk_instance,
    build_scmf_qaoa_reproduction_suite,
)


__all__ = [
    'build_qrbnbr_s1_instance',
    'build_qrbnbr_s1_suite',
    'build_scmf_gaussian_sk_instance',
    'build_scmf_qaoa_reproduction_suite',
]
