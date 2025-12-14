import time
import random

import m1_new_attack
import m2_assumption1
import m3_assumption2
import parameters
from common import get_b1_b2
from common import set_global_rng
from common import is_prime

SEED = 5
ITERATIONS = 2

if __name__ == "__main__":
    print(f"{ITERATIONS=}")
    print("\n" + "-"*40 + "(M1) Testing New Attack" + "-"*40)
    print(f"{SEED=}")
    seed_rng = random.Random(SEED)
    for p in parameters.get():
        # the m1_new_attack proof-of-concept is only suitable for small parameters
        if p.group != "small": continue
        if not is_prime(p.q): continue # prime power subfield not supported
        n = getattr(p, "n", p.m) # set n = m if n not specified
        pp = dict(q=p.q, m=p.m, n=n, k=p.k, ell1=p.ell1, ell2=p.ell2,
                  code_family="GABIDULIN",
                  seed=seed_rng.randint(0, 2**31 - 1))
        for iteration in range(ITERATIONS):
            print(pp, end=' ')
            start = time.perf_counter()
            success = m1_new_attack.run(**pp)
            end = time.perf_counter()
            elapsed = end-start
            print(dict(test='m1', iteration=iteration, success=success, elapsed=elapsed))

    print("-"*40 + "(M2) Testing Assumption 1" + "-"*40)
    print(f"{SEED=}")
    seed_rng = random.Random(SEED)
    for p in parameters.get():
        if not is_prime(p.q): continue # prime power subfield not supported
        n = getattr(p, "n", p.m) # set n = m if n not specified
        b1, b2 = get_b1_b2(k=p.k, m=p.m, n=n, ell1=p.ell1, ell2=p.ell2)
        pp = dict(q=p.q, m=p.m, n=n, k=p.k, ell1=p.ell1, b1=b1, b2=b2,
                  code_family="GABIDULIN",
                  optimize=(p.group != "small"),
                  seed=seed_rng.randint(0, 2**31 - 1))
        for iteration in range(ITERATIONS):
            print(pp, end=' ')
            start = time.perf_counter()
            success = m2_assumption1.run(**pp)
            end = time.perf_counter()
            elapsed = end-start
            print(dict(test='m2', iteration=iteration, success=success, elapsed=elapsed))

    print("\n" + "-"*40 + "(M3) Testing Assumption 2" + "-"*40)
    print(f"{SEED=}")
    seed_rng = random.Random(SEED)
    for p in parameters.get():
        if not is_prime(p.q): continue # prime power subfield not supported
        n = getattr(p, "n", p.m) # set n = m if n not specified
        pp = dict(q=p.q, m=p.m, n=n, k=p.k, ell1=p.ell1,
                  code_family="GABIDULIN",
                  optimize=(p.group != "small"),
                  seed=seed_rng.randint(0, 2**31 - 1))
        for iteration in range(ITERATIONS):
            print(pp, end=' ')
            start = time.perf_counter()
            success = m3_assumption2.run(**pp)
            end = time.perf_counter()
            elapsed = end-start
            print(dict(test='m3', iteration=iteration, success=success, elapsed=elapsed))
