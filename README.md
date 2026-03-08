# Code Supplement for ACDGV Key Attack

Main Paper: **Key Attack on the ACDGV Matrix Encryption Scheme** by *Anmoal Porwal, Antonia Wachter-Zeh, and Pierre Loidreau* (Full version: https://eprint.iacr.org/2025/1292).

This repository contains code to:
- Test 1: Run a proof-of-concept of the attack (see docstring in `m1_new_attack.py`).
- Test 2: Verify Assumption 1 (`m2_assumption1.py`).
- Test 3: Verify Assumption 2 (`m3_assumption2.py`).
- Compute complexity expressions to reproduce Table 3 in the paper (`compute_complexity.py`).

Notes:
- The code only supports extension fields with prime base fields (e.g., GF(q^m) with q=16 is not supported).
- Test 1 is implemented without optimizations for easier verifiability, and is only suitable for small parameter sets.

Three scripts can be run directly:
- **`run_tests.py`**: runs tests 1/2/3; supports `--seed`, `--iterations`, and `--tests`.
- **`run_tests_parallel.py`**: parallel runner for tests 2 and 3; suitable for large iteration counts; writes `key=value; ...` lines to timestamped output files.
- **`compute_complexity.py`**: computes complexity expressions for all ACDGV parameter sets (Table 3).

## Installation and Usage
```bash
# installation
git clone https://github.com/mnap/acdgv-key-attack.git
cd acdgv-key-attack
uv sync --locked

# usage
uv run python run_tests.py --seed 5 --iterations 2 --tests m1 m2 m3 # default
uv run python run_tests.py # same as above
uv run python run_tests.py --tests m1 # run only test 1

uv run python run_tests_parallel.py --seed 5 --iterations 50 --tests m2 m3 # default (uses half cores)
uv run python run_tests_parallel.py # same as above
uv run python run_tests_parallel.py --workers -1 # same as above but use all available cores
uv run python run_tests_parallel.py --workers 4 # same as above but use 4 cores

uv run python compute_complexity.py # compute complexity expressions (Table 3)

# The full version of the paper states that we get a 100% success rate for the two assumptions with seed=5 and iterations=50
# This can be tested with (may take many hours depending on machine):
uv run python run_tests_parallel.py --workers -1
printf "success rate: %s/%s\n" $(grep -o "success=True" benchmark*.txt | wc -l) $(grep -o "success=" benchmark*.txt | wc -l)
```

## Requirements
- Python >= 3.13
- Dependencies listed in `pyproject.toml`

## License
This project is licensed under the MIT License.