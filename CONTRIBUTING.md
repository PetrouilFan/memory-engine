# Contributing to Memory Engine

Thanks for your interest in contributing! This guide will help you get started.

## Getting Started

1. **Fork** the repository and clone your fork
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the tests** to make sure everything works:
   ```bash
   bash run_tests.sh
   ```

## Development Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run the test suite to verify nothing is broken
4. Commit with a clear, descriptive message
5. Push and open a Pull Request

## Project Layout

- **`core/`** — Core modules. Changes here affect the entire system. Be careful and add tests.
- **`pipelines/`** — Data ingestion and maintenance scripts. These are more self-contained.
- **`tools/`** — Standalone utilities.
- **`evaluation/`** — Benchmarking tools. Add new gold queries to `gold_queries.json` when adding features.
- **`tests/`** — Test suite. Always add tests for new functionality.

## Code Style

- Python 3.10+
- Use type hints for function signatures
- Follow existing patterns — look at similar code in the repo
- Keep functions focused and well-documented
- Add docstrings to new modules and classes

## Adding New Hallucination Patterns

If you discover a new class of hallucinated memories:

1. Add the regex pattern(s) to `HALLUCINATION_PATTERNS` in [core/hallucination_filter.py](core/hallucination_filter.py)
2. Add test cases to [tests/test_hallucination_filter.py](tests/test_hallucination_filter.py)
3. Document the category in [HALLUCINATION_FILTERING.md](HALLUCINATION_FILTERING.md)

## Reporting Issues

- Check existing issues first
- Include reproduction steps
- Include your Python version and OS
- Attach relevant logs if possible

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
