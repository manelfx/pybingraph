UV := uv run
PYTHON_VERSIONS ?= 3.12 3.13 3.14 3.15

# These are commands, not filesystem targets. Declaring them phony prevents
# existing directories such as `tests/` from making Make skip their recipes.
.PHONY: units format format-check lint typecheck check compatibility coverage goldens \
	goldens-checkpoint goldens-promote goldens-symbols badges badges-check python-matrix

# Run the default fast unit-test suite selected by pytest's `testpaths` setting.
units:
	$(UV) pytest -v

# Rewrite Python source and test files into Ruff's canonical formatting style.
format:
	$(UV) ruff format src tests

# Verify formatting without modifying files; suitable for review and CI.
format-check:
	$(UV) ruff format --check src tests

# Check source and tests for lint issues such as unused imports and bad patterns.
lint:
	$(UV) ruff check src tests

# Statically type-check production code. Tests are intentionally excluded
# because partial fakes and monkeypatching make their type diagnostics noisy.
typecheck:
	$(UV) ty check src

# Run the regular non-golden development checks without changing source files.
check: badges-check format-check lint typecheck units

# Generate only on request; validation is offline and never modifies the badge.
badges:
	$(UV) --no-project python scripts/badges.py $(PYTHON_VERSIONS)

badges-check:
	$(UV) --no-project python scripts/badges.py --check $(PYTHON_VERSIONS)

# The packaging job already audits 3.12; export the remaining CI matrix as JSON.
python-matrix:
	@$(UV) --no-project python -c \
	 'import json,sys; print(json.dumps({"python":sys.argv[1:]}))' \
	 $(filter-out 3.12,$(PYTHON_VERSIONS))

# Audit one wheel across interpreters without modifying .venv or build outputs.
# Requires Clang and Graphviz; override PYTHON_VERSIONS to narrow the matrix.
compatibility:
	@set -eu; \
	 audit=$$(mktemp -d "$${TMPDIR:-/tmp}/bingraph-compatibility.XXXXXX"); \
	 trap 'rm -rf "$$audit"' EXIT HUP INT TERM; \
	 $(UV) --no-project python src/bingraph/version.py --json > "$$audit/version.json"; \
	 uv build --wheel --out-dir "$$audit"; \
	 $(UV) --no-project python scripts/distributions.py \
	   --dist "$$audit" --provenance "$$audit/version.json" \
	   --python-versions $(PYTHON_VERSIONS)

# Combine unit and curated golden coverage, then write a browsable report to
# htmlcov/index.html. BINGRAPH_GOLDEN_CONFIGS remains configurable from the
# environment, just like the standalone golden targets.
coverage:
	$(UV) coverage erase
	$(UV) pytest -v --cov=bingraph --cov-append --cov-report= tests/units
	$(UV) pytest -v -m checkpoint --cov=bingraph --cov-append --cov-report= \
		tests/goldens/test_cfg_goldens.py
	$(UV) coverage report
	$(UV) coverage html

# Render and compare the complete CFG golden matrix. This can be expensive and
# writes fresh candidate artifacts under `tests/_actual/`. It also verifies
# that application symbol discovery remains aligned with the CSV corpus.
goldens: goldens-symbols
	$(UV) pytest -v --durations=5 tests/goldens/test_cfg_goldens.py

# Render and compare only checkpoint-marked goldens. The selected-test set
# keeps the full-corpus summary and non-checkpoint `_actual` artifacts intact.
# Set `BINGRAPH_GOLDEN_CONFIGS` in the environment to choose configurations.
goldens-checkpoint:
	$(UV) pytest -v --durations=5 -m checkpoint tests/goldens/test_cfg_goldens.py

# Copy complete existing `_actual` trees into `tests/goldens/` without
# rerendering CFGs. Run a full compare-mode golden suite before promoting.
goldens-promote:
	BINGRAPH_GOLDEN_MODE=promote \
		$(UV) pytest -v tests/goldens/test_cfg_goldens.py

# Compare application symbol discovery against the CSV corpus, loading one
# playground binary per test case. This is a semantic audit, not a renderer,
# so it has no promotion mode or generated graph artifacts.
goldens-symbols:
	$(UV) pytest -v --durations=5 tests/goldens/test_symbols_goldens.py
