# How to contribute to `destiny-deduper`

Last updated: NL, 2026-08-13

## Generally applicable guidelines

- Any contribution should be encapsulated within a pull request (PR), from a new branch whose sole purpose is the implementation of the contribution.
- Typically, PRs should reference issues. Sometimes it's incovenient to immediately associate a PR with an issue, but ideally the merging of a PR should close >=1 issue(s).
- We enforce [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) tags and require you to use pre-commit hooks, installed via `pre-commit install`. These tags will lead to your feature potentially incrementing the version number of `destiny-deduper`. Please keep this in mind when tagging your commits.
- In the spirit of atomicity, keep in mind the reviewer's time when putting together your PR. This should reflect both a manageable complexity and length of the new feature.
- Some people enjoy using AI-assisted coding, and that's cool. But the notion that tools like Cursor, Claude Code, Copilot etc. will __10x__ your software development chops are debateable, at best. For the purpose of contributing to `deet`, please ensure that you've self-reviewed your AI code to the degree that you're 100% sure it's the absolute best it can be before asking for review. Do _not_ throw end-to-end AI code to a human reviewer, as this simply externalises the effort onto the review process.
- __BEFORE ASKING FOR REVIEW__, please ensure the following:
  - all existing and new tests are passing, both locally and in Continuous integration (CI)
  - the core functionality of the application (i.e. the core CLI data extraction flow) is still functional locally (as we currently don't test this in CI)
  - your contribution passes linting (`ruff`) and `mypy`.
  - your contribution is well-documented, to the point that the PR summary itemises the changes you've made.
- Note that you can't expect your colleagues to include running your code in the context of reviewing it. __The onus of ensuring a) that your code works and b) that it doesn't break existing functionality is ___on you___.__
- Copilot can be a decent PR reviewer, especially before you ask a fellow contributor for a review. Copilot alone should typically not be sufficient for allowing a PR to be merged however.
- Once a PR is approved and ready for review, the original author should merge the commit into the target branch.

### PR merged into main

CI runs python-semantic-release and:

- strips the pre-release suffix to produce the canonical version (e.g. `0.2.0`)
- writes it to `pyproject.toml`
- creates git tag `0.2.0`
- creates a github release and updates `CHANGELOG.md`

Canonical versions only exist on `main`.

### Checking the version

Check pyproject.toml directly.

### Notes

- Use `uv run semantic-release version --noop` to preview what a release would do locally. Never run it without `--noop`.
- Adding the `[skip ci]` tag in actions workflows prevents the workflow from triggering itself in a loop.
- If a merge contains only `chore:`, `docs:`, or `test:` commits, no version bump or release is created.
- Do not create or move git tags manually; let CI own them entirely.
