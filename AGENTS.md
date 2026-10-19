# Agent Workflow

## Git Workflow

Read [Git](docs/git.md) before preparing changes for review, changing Git state,
or creating commits and pushes. Its checkpoint, stash ownership, user-work
protection and commit-message rules are mandatory.

## Python Tooling

- Run Python commands, tests, linters, and formatters through `uv run`.
- Use the repository Makefile targets where they cover the requested check.
- Run focused tests first; do not run the full golden corpus unless requested.

Read [Testing](docs/testing.md) before planning validation, changing CFG behavior,
or working with golden artifacts. It contains the mandatory golden-ownership,
checkpoint-validation and promotion rules, plus compatibility and CI guidance.
The Makefile and workflow files are authoritative for commands and execution
order.

## Source Organization

Read the [source map](docs/development.md#source-organization) when moving modules
or changing CFG responsibilities. Preserve resolver priority, proof budgets,
cache lifetime and unknown-target handling. Presentation must not change control
flow; recovery hints must not become exact proof facts.

Use [Development](docs/development.md#dependency-changes) when updating dependency
bounds or lockfiles. Update the relevant guide in the same change when
responsibilities or procedures change, without weakening the user's workflow
rules. Do not use these documents as development logs; future work belongs in
`TODO`.

## Versioning and Releases

Read [Releases](docs/releases.md) before changing versioning, packaging or
publication workflows, or preparing a release. Its version/provenance invariants,
asset-preservation and approval rules are mandatory; the guide also covers draft
attachments, containers and PyPI configuration.
