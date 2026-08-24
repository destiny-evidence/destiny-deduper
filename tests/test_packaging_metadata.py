import tomllib
from pathlib import Path


def _dependency_names(dependencies: list[str]) -> set[str]:
    return {dependency.split(";", 1)[0].split("[", 1)[0].split(">=", 1)[0] for dependency in dependencies}


def test_release_tooling_is_not_a_runtime_dependency() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    runtime_dependencies = _dependency_names(pyproject["project"]["dependencies"])
    dev_dependencies = _dependency_names(pyproject["dependency-groups"]["dev"])

    assert "python-semantic-release" not in runtime_dependencies
    assert "python-semantic-release" in dev_dependencies
