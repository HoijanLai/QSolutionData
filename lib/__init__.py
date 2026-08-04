"""Public QSolutionData API with dependency-safe lazy exports.

Historically this package imported preprocessing, pipeline, portfolio, compiler,
and adapter modules as soon as *any* ``lib`` submodule was imported.  That made
lightweight code such as ``import lib.solvers.qubo`` load pandas and every
portfolio dependency before a solver could run.

PEP 562 module ``__getattr__`` keeps the same public names while importing only
the module that owns the requested symbol.  Access remains source-compatible:
``from lib import compile_qubo`` still returns the same function object.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    'asset_classification': (
        '.preprocessing',
        'asset_classification',
    ),
    'asset_scoring': (
        '.preprocessing',
        'asset_scoring',
    ),
    'asset_similarity': (
        '.preprocessing',
        'asset_similarity',
    ),
    'build_portfolio_cbqm': (
        '.portfolio',
        'build_portfolio_cbqm',
    ),
    'client_profile': (
        '.portfolio',
        'client_profile',
    ),
    'CbqmSolver': (
        '.contracts',
        'CbqmSolver',
    ),
    'compile_qubo': (
        '.compilers',
        'compile_qubo',
    ),
    'portfolio_constraints': (
        '.portfolio',
        'portfolio_constraints',
    ),
    'prepare_qubo_inputs': (
        '.pipeline',
        'prepare_qubo_inputs',
    ),
    'QiskitQuadraticProgramAdapter': (
        '.adapters',
        'QiskitQuadraticProgramAdapter',
    ),
    'MisSolver': (
        '.contracts',
        'MisSolver',
    ),
    'QuboSolver': (
        '.contracts',
        'QuboSolver',
    ),
    'Sampler': (
        '.contracts',
        'Sampler',
    ),
    'QRBnBRMaxCutAdapter': (
        '.adapters',
        'QRBnBRMaxCutAdapter',
    ),
    'QRBnBRSolverAdapter': (
        '.adapters',
        'QRBnBRSolverAdapter',
    ),
    'Solver': (
        '.contracts',
        'Solver',
    ),
    'weight_encoding': (
        '.portfolio',
        'weight_encoding',
    ),
}

# These package attributes were historically available through normal Python
# package import behavior (for example ``from lib import solvers``).  Handling
# them explicitly makes that behavior deterministic without adding them to the
# established star-import surface below.
_LAZY_SUBMODULES = {
    'adapters',
    'compilers',
    'contracts',
    'pipeline',
    'portfolio',
    'preprocessing',
    'solvers',
}

__all__ = [
    'asset_classification',
    'asset_scoring',
    'asset_similarity',
    'build_portfolio_cbqm',
    'client_profile',
    'CbqmSolver',
    'compile_qubo',
    'portfolio_constraints',
    'prepare_qubo_inputs',
    'QiskitQuadraticProgramAdapter',
    'MisSolver',
    'QuboSolver',
    'QRBnBRMaxCutAdapter',
    'QRBnBRSolverAdapter',
    'Sampler',
    'Solver',
    'weight_encoding',
]


def __getattr__(name):
    """Resolve one public symbol or subpackage on first access."""
    export = _LAZY_EXPORTS.get(name)
    if export is not None:
        module_name, attribute_name = export
        value = getattr(import_module(module_name, __name__), attribute_name)
        globals()[name] = value
        return value

    if name in _LAZY_SUBMODULES:
        value = import_module(f'.{name}', __name__)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Expose lazy names to IDEs and interactive discovery."""
    return sorted(
        set(globals())
        | set(_LAZY_EXPORTS)
        | _LAZY_SUBMODULES
    )
