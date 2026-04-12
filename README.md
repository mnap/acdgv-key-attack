# Code Supplement for ACDGV Key Attack

Main Paper: **Key Attack on the ACDGV Matrix Encryption Scheme** by *Anmoal Porwal, Antonia Wachter-Zeh, and Pierre Loidreau* (Full version: https://eprint.iacr.org/2025/1292).

This repository contains code to:
- Test 1: Run a proof-of-concept of the key-recovery attack and check whether the derived matrix code is equivalent to the secret matrix code (`m1_new_attack.py`).
- Test 2: Verify Assumption 1 (`m2_assumption1.py`).
- Test 3: Verify Assumption 2 (`m3_assumption2.py`).
- Compute complexity expressions to reproduce Table 3 in the paper (`compute_complexity.py`).

The main scripts intended to be run directly are:
- **`run_tests.py`**: runs tests 1, 2 and 3; supports `--seed`, `--iterations`, `--tests` and `--max_m` (see Usage section below).
- **`run_tests_parallel.py`**: parallel runner for tests 2 and 3; suitable for large iteration counts; writes results to timestamped output files.
- **`compute_complexity.py`**: computes complexity expressions for all ACDGV parameter sets (Table 3).

Notes:
- All the parameters are listed in `parameters.py`.
- The code only supports extension fields with prime base fields (e.g., GF(q^m) with q=16 is not supported).
- Test 1 is implemented without optimizations for easier verifiability and is only suitable for small parameter sets.
- The underlying implementations for tests 2 and 3 both take a boolean `optimize` parameter. If `optimize=True` the code uses some tricks to run faster but the downside is that the code is harder to verify.

Therefore, both `run_tests.py` and `run_tests_parallel.py` (1) exclude parameters when q is not prime, (2) only run the `group=small` parameters for test 1, and (3) set `optimize=True` for all parameter sets except when `group=small` for tests 2 and 3.

## Installation
```bash
git clone https://github.com/mnap/acdgv-key-attack.git
cd acdgv-key-attack
```
With `uv` (tested with version 0.6.6):
```bash
uv sync --locked
```

(Alternative) With Docker:
```bash
docker build -t acdgv-key-attack-image .
docker run --rm -it --mount type=bind,source="$(pwd)",target=/workspace acdgv-key-attack-image
```
The Dockerfile and `docker-entrypoint.sh` were kindly provided by a reviewer and adapted slightly.
If you want to set up the Python environment manually, then the required dependencies are listed in `pyproject.toml`.

## Usage
If you installed using Docker, replace `uv run python` below with `python`.

```bash
uv run python run_tests.py --seed 5 --iterations 2 --tests m1 m2 m3 # default
uv run python run_tests.py # same as above
uv run python run_tests.py --tests m1 # run only test 1

uv run python run_tests_parallel.py --seed 5 --iterations 50 --tests m2 m3 # default (uses half cores)
uv run python run_tests_parallel.py # same as above
uv run python run_tests_parallel.py --workers -1 # same as above but use all available cores
uv run python run_tests_parallel.py --workers 4 # same as above but use 4 cores

uv run python compute_complexity.py # compute complexity expressions (Table 3)

# The full version of the paper states that we get a 100% success rate for the two assumptions with seed=5 and iterations=50
# This can be tested with (may take many hours or even days depending on machine):
uv run python run_tests_parallel.py --workers -1
printf "success rate: %s/%s\n" $(grep -o "success=True" benchmark*.txt | wc -l) $(grep -o "success=" benchmark*.txt | wc -l)
# Note: the largest parameter set with m=79 takes a very long time to run
# Use --max_m=X to test only parameter sets with m <= X. Hence, to exclude the m=79 set, run
uv run python run_tests_parallel.py --workers -1 --max_m=78
```

## License
This project is licensed under the MIT License.
