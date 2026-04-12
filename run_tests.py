import time
import random
import argparse

import m1_new_attack
import m2_assumption1
import m3_assumption2
import parameters
from common import get_b1_b2
from common import is_prime

BASE_SEED = 5
ITERATIONS = 2
TESTS = ("m1", "m2", "m3")  # default tests to run


def parse_args():
    parser = argparse.ArgumentParser(description="Run ACDGV key-attack supplemental tests.")
    parser.add_argument("--seed", type=int, default=BASE_SEED,
                        help="Base seed (default: %(default)s)")
    parser.add_argument("--iterations", type=int, default=ITERATIONS,
                        help="Iterations per parameter set (default: %(default)s)")
    parser.add_argument("--max_m", type=int, default=None,
                        help="Skip parameter sets with m > max_m")
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=("m1", "m2", "m3"),
        default=list(TESTS),
        help="Which tests to run (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    BASE_SEED = args.seed
    ITERATIONS = args.iterations
    TESTS = set(args.tests)
    MAX_M = args.max_m

    print(f"{ITERATIONS=}")
    print(f"{BASE_SEED=}")
    print(f"{MAX_M=}")
    print(f"TESTS={sorted(TESTS)}")

    if "m1" in TESTS:
        print("\n" + "-"*40 + "(M1) Testing New Attack" + "-"*40)
        params_list = [
            p for p in parameters.get()
            if p.group == "small" and is_prime(p.q)
        ]
        seed_rng = random.Random(BASE_SEED)
        all_seeds = [seed_rng.randint(0, 2**31 - 1) for _ in range(len(params_list) * ITERATIONS)]
        seed_idx = 0
        for p in params_list:
            n = getattr(p, "n", p.m)
            for iteration in range(ITERATIONS):
                seed = all_seeds[seed_idx]
                seed_idx += 1
                if MAX_M is not None and p.m > MAX_M:
                    continue
                pp = dict(q=p.q, m=p.m, n=n, k=p.k, ell1=p.ell1, ell2=p.ell2,
                          code_family="GABIDULIN",
                          seed=seed)
                print(pp, end=' ')
                start = time.perf_counter()
                success = m1_new_attack.run(**pp)
                elapsed = time.perf_counter() - start
                print("result=",
                    dict(test='m1', iteration=iteration, success=success, elapsed=elapsed),
                    sep="")
                print()

    if "m2" in TESTS:
        print("-"*40 + "(M2) Testing Assumption 1" + "-"*40)
        params_list = [p for p in parameters.get() if is_prime(p.q)]
        seed_rng = random.Random(BASE_SEED)
        all_seeds = [seed_rng.randint(0, 2**31 - 1) for _ in range(len(params_list) * ITERATIONS)]
        seed_idx = 0
        for p in params_list:
            n = getattr(p, "n", p.m)
            b1, b2 = get_b1_b2(k=p.k, m=p.m, n=n, ell1=p.ell1, ell2=p.ell2)
            for iteration in range(ITERATIONS):
                seed = all_seeds[seed_idx]
                seed_idx += 1
                if MAX_M is not None and p.m > MAX_M:
                    continue
                pp = dict(q=p.q, m=p.m, n=n, k=p.k, ell1=p.ell1, b1=b1, b2=b2,
                          code_family="GABIDULIN",
                          optimize=(p.group != "small"),
                          seed=seed)
                print(pp, end=' ')
                start = time.perf_counter()
                success = m2_assumption1.run(**pp)
                elapsed = time.perf_counter() - start
                print("result=",
                    dict(test='m2', iteration=iteration, success=success, elapsed=elapsed),
                    sep="")

    if "m3" in TESTS:
        print("\n" + "-"*40 + "(M3) Testing Assumption 2" + "-"*40)
        params_list = [p for p in parameters.get() if is_prime(p.q)]
        seed_rng = random.Random(BASE_SEED)
        all_seeds = [seed_rng.randint(0, 2**31 - 1) for _ in range(len(params_list) * ITERATIONS)]
        seed_idx = 0
        for p in params_list:
            n = getattr(p, "n", p.m)
            for iteration in range(ITERATIONS):
                seed = all_seeds[seed_idx]
                seed_idx += 1
                if MAX_M is not None and p.m > MAX_M:
                    continue
                pp = dict(q=p.q, m=p.m, n=n, k=p.k, ell1=p.ell1,
                          code_family="GABIDULIN",
                          optimize=(p.group != "small"),
                          seed=seed)
                print(pp, end=' ')
                start = time.perf_counter()
                success = m3_assumption2.run(**pp)
                elapsed = time.perf_counter() - start
                print("result=",
                      dict(test='m3', iteration=iteration, success=success, elapsed=elapsed),
                      sep="")
