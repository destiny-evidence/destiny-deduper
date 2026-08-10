"""Init file for algorithm sub-module."""

import importlib.util
import warnings

_REQUIRED_FOR_ALGORITHM = {
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "tqdm": "tqdm",
}

_missing = [
    package_name
    for module_name, package_name in _REQUIRED_FOR_ALGORITHM.items()
    if importlib.util.find_spec(module_name) is None
]

if _missing:
    warnings.warn(
        "Algorithm modules may not work fully. Missing optional dependencies: "
        + ", ".join(sorted(_missing))
        + ". Install with: uv sync --group algorithm-and-notebooks",
        category=UserWarning,
        stacklevel=2,
    )
