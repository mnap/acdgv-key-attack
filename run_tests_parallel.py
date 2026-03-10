"""
Parallel script for running m2 and/or m3 tests with multiple iterations.
Writes results to an output file in `key=value; ...` format.
"""
import os
import time
import random
import argparse
from multiprocessing import Pool
from datetime import datetime

import m2_assumption1
import m3_assumption2
import parameters
from common import get_b1_b2
from common import is_prime


def run_single_m2_test(args):
    """Run a single m2 test iteration. Returns a result dict."""
    params, iteration, seed = args
    p = params
    n = getattr(p, "n", p.m)
    b1, b2 = get_b1_b2(k=p.k, m=p.m, n=n, ell1=p.ell1, ell2=p.ell2)
    pp = dict(q=p.q, m=p.m, n=n, k=p.k, ell1=p.ell1, b1=b1, b2=b2, code_family="GABIDULIN",
        optimize=(p.group != "small"), seed=seed)
    start = time.perf_counter()
    success = m2_assumption1.run(**pp)
    elapsed = time.perf_counter() - start
    return pp | {"test": "m2", "iteration": iteration, "group": p.group, "success": success,
        "elapsed": elapsed, "timestamp": datetime.now().isoformat(),}


def run_single_m3_test(args):
    """Run a single m3 test iteration. Returns a result dict."""
    params, iteration, seed = args
    p = params
    n = getattr(p, "n", p.m)
    pp = dict(q=p.q, m=p.m, n=n, k=p.k, ell1=p.ell1, code_family="GABIDULIN",
        optimize=(p.group != "small"), seed=seed)
    start = time.perf_counter()
    success = m3_assumption2.run(**pp)
    elapsed = time.perf_counter() - start
    return pp | {"test": "m3", "iteration": iteration, "group": p.group, "success": success,
        "elapsed": elapsed, "timestamp": datetime.now().isoformat(),}


def append_single_result(result, filename):
    """Append a result dictionary as a single `key=value; ...` line to filename."""
    if not result:
        return
    result_line = "; ".join(f"{k}={v}" for k, v in result.items())
    print(result_line)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(result_line + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel benchmark runner for m2 (Assumption 1) and m3 (Assumption 2)."
    )
    parser.add_argument("--seed", type=int, default=5, help="Base seed (default: %(default)s)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Iterations per parameter set (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of worker processes: \
                        -1=all cores, 0=half cores (default), N=exact.")
    parser.add_argument("--tests", nargs="+", choices=("m2", "m3"), default=["m2", "m3"],
                        help="Which tests to run (default: %(default)s)",)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.workers < -1:
        parser.error("--workers must be -1, 0, or a positive integer")
    return args


if __name__ == "__main__":
    args = parse_args()
    BASE_SEED = args.seed
    ITERATIONS = args.iterations
    tests = set(args.tests)

    total_cores = os.cpu_count() or 1
    half_cores = max(1, total_cores // 2)
    if args.workers == -1:
        workers = total_cores
    elif args.workers == 0:
        workers = half_cores
    else:
        if args.workers < 1:
            raise ValueError("--workers must be -1, 0, or a positive integer")
        workers = args.workers

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    outputfile_m2 = f"benchmark_m2_results_{timestamp_str}.txt"
    outputfile_m3 = f"benchmark_m3_results_{timestamp_str}.txt"

    if "m2" in tests and os.path.exists(outputfile_m2):
        raise RuntimeError(f"File {outputfile_m2} already exists.")
    if "m3" in tests and os.path.exists(outputfile_m3):
        raise RuntimeError(f"File {outputfile_m3} already exists.")

    print(f"Using {workers} out of {total_cores} parallel workers")
    print(f"Base seed: {BASE_SEED}")
    print(f"Iterations per parameter set: {ITERATIONS}")
    print(f"Tests: {sorted(tests)}")
    if "m2" in tests:
        print(f"Output file (m2): {outputfile_m2}")
    if "m3" in tests:
        print(f"Output file (m3): {outputfile_m3}")
    print()

    params_list = [p for p in parameters.get() if is_prime(p.q)]
    print(f"Found {len(params_list)} parameter sets to test (prime q only)")

    total_tests = len(params_list) * ITERATIONS
    print(f"Total number of test runs per test: {total_tests}")
    print(f"Generating {total_tests} deterministic seeds from base seed {BASE_SEED}...")
    seed_rng = random.Random(BASE_SEED)
    all_seeds = [seed_rng.randint(0, 2**31 - 1) for _ in range(total_tests)]

    # Optional: add a short header to outputs
    header = f"# base_seed={BASE_SEED}; iterations={ITERATIONS}; workers={workers}; timestamp={timestamp_str}"
    if "m2" in tests:
        with open(outputfile_m2, "a", encoding="utf-8") as f:
            f.write(header + "\n")
    if "m3" in tests:
        with open(outputfile_m3, "a", encoding="utf-8") as f:
            f.write(header + "\n")

    if "m2" in tests:
        print(f"\n{'='*60}")
        print("Running m2 (Assumption 1) tests...")
        print(f"{'='*60}")
        m2_args = []
        seed_idx = 0
        for p in params_list:
            for i in range(ITERATIONS):
                m2_args.append((p, i, all_seeds[seed_idx]))
                seed_idx += 1
        print(f"Total m2 test runs: {len(m2_args)}")
        start_time = time.perf_counter()
        with Pool(workers) as pool:
            print("Starting m2 tests with immediate result saving...")
            for result in pool.imap_unordered(run_single_m2_test, m2_args):
                append_single_result(result, outputfile_m2)
        elapsed = time.perf_counter() - start_time
        print(f"Completed m2 tests in {elapsed:.2f} seconds")
        print(f"Average time per test: {elapsed/len(m2_args):.3f} seconds")

    if "m3" in tests:
        print(f"\n{'='*60}")
        print("Running m3 (Assumption 2) tests...")
        print(f"{'='*60}")
        m3_args = []
        seed_idx = 0  # reset so m3 uses the same seed stream as m2
        for p in params_list:
            for i in range(ITERATIONS):
                m3_args.append((p, i, all_seeds[seed_idx]))
                seed_idx += 1
        print(f"Total m3 test runs: {len(m3_args)}")
        start_time = time.perf_counter()
        with Pool(workers) as pool:
            print("Starting m3 tests with immediate result saving...")
            for result in pool.imap_unordered(run_single_m3_test, m3_args):
                append_single_result(result, outputfile_m3)
        elapsed = time.perf_counter() - start_time
        print(f"Completed m3 tests in {elapsed:.2f} seconds")
        print(f"Average time per test: {elapsed/len(m3_args):.3f} seconds")
