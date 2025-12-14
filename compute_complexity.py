import traceback # for debugging

from numpy import log2

from common import get_b1_b2
import parameters


if __name__ == '__main__':
    param_list = parameters.get()
    filtered_param_list = [x for x in param_list if getattr(x, "group", None) in ["acdgv_fig5", "acdgv_fig6"]]
    # sort w.r.t. pk_size
    filtered_param_list = sorted(
        filtered_param_list,
        key=lambda p: (p.design,
                       p.k*p.m*((p.m+p.ell1)*(p.m+p.ell2) - p.k*p.m)*log2(p.q))
    )
    try:
        last_design = None
        for p in filtered_param_list:
            wf = {128: 143, 192: 207, 256: 272}.get(p.design)
            q, k, m, ell1, ell2, r = p.q, p.k, p.m, p.ell1, p.ell2, p.r
            n = getattr(p, "n", p.m)
            if m != n:
                print("Warning: m != n. The computation of polynomial factors will be incorrect.")
            R = k/m
            b1, b2 = get_b1_b2(k=k, m=m, n=n, ell1=ell1, ell2=ell2)
            assert b1 <= m and b2 <= n
            assert (m+ell1)/(k*m) <= 1 # epsilon <= 1
            cmp_new_exp = log2(q)*(b1*ell1+b2*ell2)
            cmp_new_poly = log2(log2(q)) + 8*log2(m)
            cmp_naive_srch_exp = (m*ell1 + n*ell2)*log2(q)
            cmp_naive_srch_poly = log2(log2(q)) + 6*log2(m)
            cmp_naive_dstg_exp = (m*ell1 + (k+1)*ell2)*log2(q)
            cmp_naive_dstg_poly = log2(log2(q)) + (7)*log2(m)
            vals_exp = [cmp_new_exp, cmp_naive_dstg_exp, cmp_naive_srch_exp]
            vals_pol = [cmp_new_poly, cmp_naive_dstg_poly, cmp_naive_srch_poly]
            pk_size_kB = k*m*((m+ell1)*(m+ell2) - k*m)*log2(q)/(1000*8)
            cp_size_B = ((m+ell1)*(m+ell2) - k*m)*log2(q)/8
            vals = [x + y for x,y in zip(vals_exp, vals_pol)]

            if last_design != p.design:
                print()
                print(f"Security level: {wf} bits (AES-{p.design})")
                print("-" * 120)
                print(
                    f"{'q':>6} {'k/m':>6} {'m':>5} {'ell1':>5} {'ell2':>5} "
                    f"{'pk[kB]':>8} {'cp[B]':>8} "
                    f"{'new':>8} {'acdgv':>8} {'naive':>8}"
                )
                print("-" * 120)

            min_val = min(vals)
            rounded = [int(f"{v:.0f}") for v in vals]

            def fmt(v):
                s = str(v)
                if v == int(f"{min_val:.0f}"):
                    s += "*"
                if v <= wf:
                    s = f"[{s}]"
                return f"{s:>8}"

            print(
                f"{q:>6} {k/m:>6.2f} {m:>5} {ell1:>5} {ell2:>5} "
                f"{pk_size_kB:>8.0f} {cp_size_B:>8.0f} "
                f"{fmt(rounded[0])}{fmt(rounded[1])}{fmt(rounded[2])}"
            )
            last_design = p.design
    except Exception:
        traceback.print_exc()
