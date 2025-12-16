# Code Supplement for ACDGV Key Attack

## Overview

Main Paper: "Key Attack on the ACDGV Matrix Encryption Scheme".

This repository contains code to:
- Test 1: Run a proof-of-concept of the attack (see docstring in `m1_new_attack.py`).
- Test 2: Verify Assumption 1.
- Test 3: Verify Assumption 2.
- Compute complexity expressions to reproduce Table 3 in paper.

Two scripts can be run directly:

**`run_tests.py`**
Runs the proof-of-concept attack and verifies Assumptions 1 and 2 using default parameters from `parameters.py`. The script is easily modifiable, and the values of SEED and ITERATIONS can be set there.

**`compute_complexity.py`**
Computes complexity expressions for all ACDGV parameter sets.

## Installation
```bash
git clone https://github.com/mnap/acdgv-key-attack.git
cd acdgv-key-attack
# Recommended: Install with uv (uses locked dependency versions for reproducibility)
uv sync
# Alternative: Install with pip
pip install --user .
```

## Usage
```bash
# Run the proof-of-concept attack and verify Assumptions 1 and 2
python run_tests.py
# Compute complexity for all ACDGV parameters
python compute_complexity.py
```
## Requirements
- Python >= 3.13
- Dependencies listed in `pyproject.toml`

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
