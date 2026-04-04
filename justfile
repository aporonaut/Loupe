# Loupe development commands

# Format code
format:
    ruff format .

# Lint code
lint:
    ruff check .

# Type check
typecheck:
    uv run pyright src/

# Run tests
test:
    uv run pytest

# Run all verification steps
verify: format lint typecheck test
