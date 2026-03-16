# Contributing to ForceInferencePy

Thank you for considering contributing! All contributions are welcome — bug reports, documentation fixes, new tests, and new features.

---

## Getting started

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/<your-username>/ForceInferencePy.git
cd ForceInferencePy

# 2. Create a virtual environment and install in editable mode with dev tools
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 3. Verify everything works
pytest tests/
```

---

## Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout -b fix/descriptive-name
   ```
2. **Make your changes** following the code style guidelines below.
3. **Add or update tests** for any logic you change (see `tests/`).
4. **Run the test suite** locally before pushing:
   ```bash
   pytest tests/ --cov=force_inference
   ```
5. **Open a pull request** against `main` with a clear description of what changed and why.

---

## Code style

- Python ≥ 3.8 compatibility is required. Avoid `X | Y` union syntax and built-in generic aliases (`list[int]`) — use `from __future__ import annotations` or `typing` equivalents instead.
- Docstrings follow **NumPy style** for public functions.
- Line length ≤ 100 characters (enforced by `ruff`).
- Run `ruff check force_inference/` before committing; fix any reported issues.

---

## Reporting bugs

Please open a [GitHub issue](https://github.com/weiyuankong/ForceInferencePy/issues) and include:

- Python version and OS.
- Minimal reproducible example (ideally with a synthetic label image, not a real data file).
- Full traceback if an exception is raised.

---

## Code of conduct

This project follows the [Contributor Covenant Code of Conduct v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Be kind and constructive.
