# deduplication-toolkit

A suite of tools for deduplicating DESTINY `References`

## tl, dr

Often, we want to ensure that a DESTINY `Work` or `Enhancement` is not a duplicate of one that already exists in the DESTINY repository, or, more likely, a set of duplicate candidates. Hence the goal of this `deduplication-toolkit` is to provide a set of portable, customisable solutions for quantifying the likelihood that a given `Work` is a duplicate of another.

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
