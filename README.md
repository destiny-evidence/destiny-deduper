# deduplication-toolkit

A suite of tools for deduplicating DESTINY `References`

## tl, dr

Often, we want to ensure that a DESTINY `Reference` (or its sub-types) is not a duplicate of one that already exists in the DESTINY repository, or, more in some other bag of potential, pre-selected duplicate candidates. Hence the goal of this `deduplication-toolkit` is to provide a set of portable, customisable solutions for quantifying the likelihood that a given `Reference` is a duplicate of another.

## Setup

### Requirements

[uv](https://docs.astral.sh/uv) is used for dependency management and managing virtual environments. You can install uv either using pipx or the uv installer script:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installing Dependencies

Once uv is installed, install dependencies:

```sh
uv sync
```

### Activate your environment

```sh
source .venv/bin/activate
```

## Contributing

If you want to contribute to this project -- awesome, everyone's welcome.
Please see the [contributing guidelines](CONTRIBUTING.md) for details on how best to contribute.

A few important steps when contributing:

```sh
pre-commit install
uv run pre-commit install --hook-type commit-msg
```

This will force you to use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/).

### Commit message format

All commits must use [**conventional commits**](conventionalcommits.org). The pre-commit hook will reject any commit that doesn't.

| prefix | example | version effect |
|---|---|---|
| `fix:` | `fix: handle null input` | patch: `0.1.0 -> 0.1.1` |
| `feat:` | `feat: add login page` | minor: `0.1.0 -> 0.2.0` |
| `feat!:` or `BREAKING CHANGE:` footer | `feat!: remove legacy api` | major: `0.1.0 -> 1.0.0` |
| `chore:`, `docs:`, `ci:`, `test:` | `chore: update deps` | no bump |

we will udpate some stuff here!
