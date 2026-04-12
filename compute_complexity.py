import traceback  # for debugging

from numpy import log2

from common import get_b1_b2
import parameters


def compute_row_data(p):
    """
    Compute all values needed to print one table row.
    Returns a dictionary with raw + derived values.
    """
    wf = {128: 143, 192: 207, 256: 272}.get(p.design)

    q, k, m, ell1, ell2, r = p.q, p.k, p.m, p.ell1, p.ell2, p.r
    n = getattr(p, "n", p.m)

    R = k / m
    b1, b2 = get_b1_b2(k=k, m=m, n=n, ell1=ell1, ell2=ell2)

    assert b1 <= m and b2 <= n
    assert (m + ell1) / (k * m) <= 1  # epsilon <= 1
    assert m == n

    cmp_new_exp = log2(q) * (b1 * ell1 + b2 * ell2)
    cmp_naive_srch_exp = (m * ell1 + n * ell2) * log2(q)
    cmp_naive_dstg_exp = (m * ell1 + (k + 1) * ell2) * log2(q)
    cmp_new_poly = log2(k * (m - k) * m**6 * log2(q))
    cmp_naive_srch_poly = log2(log2(q)) + 6 * log2(m)
    cmp_naive_dstg_poly = log2(log2(q)) + 6 * log2(m)

    vals_exp = [cmp_new_exp, cmp_naive_dstg_exp, cmp_naive_srch_exp]
    vals_pol = [cmp_new_poly, cmp_naive_dstg_poly, cmp_naive_srch_poly]
    vals = [x + y for x, y in zip(vals_exp, vals_pol)]

    # note the Niederreiter variant of the ACDGV scheme is assumed
    # the formulas for the public key and ciphertext sizes are given in Section 7.1
    # of the ACDGV paper (https://arxiv.org/abs/2405.16539)
    pk_size_kB = k * m * ((m + ell1) * (m + ell2) - k * m) * log2(q) / (1000 * 8)
    cp_size_B = ((m + ell1) * (m + ell2) - k * m) * log2(q) / 8

    min_val = min(vals)
    rounded_vals = [int(f"{v:.0f}") for v in vals]
    min_rounded = int(f"{min_val:.0f}")

    return {
        "design": p.design,
        "wf": wf,
        "q": q,
        "k": k,
        "m": m,
        "n": n,
        "ell1": ell1,
        "ell2": ell2,
        "r": r,
        "R": R,
        "b1": b1,
        "b2": b2,
        "pk_size_kB": pk_size_kB,
        "cp_size_B": cp_size_B,
        "vals_exp": vals_exp,
        "vals_pol": vals_pol,
        "vals": vals,
        "rounded_vals": rounded_vals,   # [new, acdgv, naive]
        "min_rounded": min_rounded,
    }


if __name__ == '__main__':
    param_list = parameters.get()
    filtered_param_list = [
        x for x in param_list
        if getattr(x, "group", None) in ["acdgv_fig5", "acdgv_fig6"]
    ]

    # sort w.r.t. pk_size
    filtered_param_list = sorted(
        filtered_param_list,
        key=lambda p: (
            p.design,
            p.k * p.m * ((p.m + p.ell1) * (p.m + p.ell2) - p.k * p.m) * log2(p.q)
        )
    )

    try:
        last_design = None

        for p in filtered_param_list:
            row = compute_row_data(p)

            if last_design != row["design"]:
                print()
                print(f"Security level: {row['wf']} bits (AES-{row['design']})")
                print("-" * 120)
                print(
                    f"{'q':>6} {'k/m':>6} {'m':>5} {'ell1':>5} {'ell2':>5} "
                    f"{'pk[kB]':>8} {'cp[B]':>8} "
                    f"{'new':>8} {'acdgv':>8} {'naive':>8}"
                )
                print("-" * 120)

            def fmt(v):
                s = str(v)
                if v == row["min_rounded"]:
                    s += "*"
                if v <= row["wf"]:
                    s = f"[{s}]"
                return f"{s:>8}"

            print(
                f"{row['q']:>6} {row['R']:>6.2f} {row['m']:>5} {row['ell1']:>5} {row['ell2']:>5} "
                f"{row['pk_size_kB']:>8.0f} {row['cp_size_B']:>8.0f} "
                f"{fmt(row['rounded_vals'][0])}{fmt(row['rounded_vals'][1])}{fmt(row['rounded_vals'][2])}"
            )

            last_design = row["design"]

    except Exception:
        traceback.print_exc()
