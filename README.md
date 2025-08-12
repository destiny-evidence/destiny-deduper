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
