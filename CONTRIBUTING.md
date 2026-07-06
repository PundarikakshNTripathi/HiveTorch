# Contributing to HiveTorch

Thank you for your interest in contributing to HiveTorch! We welcome pull requests, bug reports, and feature requests.

## Development Workflow
1. Fork the repository and clone it locally.
2. Create a feature branch from `develop` following our naming convention (`feature/...`, `fix/...`, `docs/...`, `chore/...`, `perf/...`).
3. Set up the local environment by running `make setup` and `make dvc-init`.
4. Write your code and ensure you add unit/integration tests in `tests/`.
5. Run `make lint` and `make test` before committing.
6. Commit your changes using conventional commits (e.g., `feat: added new aggregation strategy`).
7. Push to your fork and submit a Pull Request against our `develop` branch.

## Code of Conduct
Please be respectful and constructive when reviewing code and interacting with other contributors.
