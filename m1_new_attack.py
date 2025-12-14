import numpy as np
import galois

from common import get_b1_b2
from common import add_random_rows_columns
from common import get_gabidulin_generator_matrix
from common import get_random_full_rank_Fq_vector
from common import get_random_full_rank_matrix
from common import get_AXB
from common import get_matrix_code_expanded_using_power_basis_of_Fqm
from common import set_global_rng


def run(q, m, n, k, ell1, ell2, code_family, seed, suppress_output=False, draw_random_VW=False):
    """Runs a proof-of-concept of one iteration of the new proposed key recovery attack. It draws a
    random code from the code_family, expands it to a matrix code Cmat and then applies the hiding
    transform to obtain Cpub. It then picks the matrices V and W as in the success condition, then
    derives a matrix code from Cpub using only V and W, and returns True if the derived matrix code
    is equivalent to the secret code Cmat.

    This code is implemented without any optimizations whatsoever and is only suitable for small
    parameters. The upside is that the code is easier to verify for correctness.

    If code_family="RANDOM" (resp. "GABIDULIN") then a random F_{q^m}-linear (Gabidulin) code is
    used. If draw_random_VW = True, then V and W are drawn randomly. Hence this will almost
    certainly fail unless the paramters are very small or if this function is called many many many
    times. If suppress_output=True, then do not print anything. The seed argument is for
    reproducibility.
    """
    set_global_rng(seed)
    K = k*m
    Fqm = galois.GF(q, m)
    Fq = Fqm.prime_subfield
    # use b1, b2 as specified in the new recovery algorithm
    b1, b2 = get_b1_b2(k=k, m=m, n=n, ell1=ell1, ell2=ell2)

    # get a basis A_list of Cmat and then derive a basis B_list of Cpub
    if code_family == "GABIDULIN":
        g = get_random_full_rank_Fq_vector(Fqm, n)
        G_Cvec = get_gabidulin_generator_matrix(g, k)
    elif code_family == "RANDOM":
        G_Cvec = get_random_full_rank_matrix(Fqm, k, n)
    else:
        raise ValueError("code_family should be either GABIDULIN or RANDOM.")
    # shape (km, m, n)
    A_list = get_matrix_code_expanded_using_power_basis_of_Fqm(G_Cvec, random_basis=True)
    B_list_pre = add_random_rows_columns(A_list, ell1, ell2)  # shape (km, m+ell1, n+ell2)
    P = get_random_full_rank_matrix(Fq, m+ell1, m+ell1)
    Q = get_random_full_rank_matrix(Fq, n+ell2, n+ell2)
    B_list = Fq.Zeros(B_list_pre.shape)
    for i in range(K):
        B_list[i] = P @ B_list_pre[i] @ Q

    V_true = P[:,m:].left_null_space().row_reduce().T # shape (m+ell1)xm
    W_true = Q[n:,:].null_space().row_reduce().T # shape (n+ell2)xn
    if draw_random_VW:
        V = get_random_full_rank_matrix(Fq, m+ell1, b1)
        W = get_random_full_rank_matrix(Fq, n+ell2, b2)
    else:
        V = V_true @ get_random_full_rank_matrix(Fq, m, b1)
        W = W_true @ get_random_full_rank_matrix(Fq, n, b2)

    # set up the MinRank-like equation and solve it:
    # FB_s - \sum muis B_i = 0 for s = 1,...,K
    VtBiW_list = Fq.Zeros((K, b1, b2))
    for i in range(K): VtBiW_list[i] = V.T @ B_list[i] @ W
    E = Fq.Zeros((K*b1*b2, (m+ell1)**2 + K*K))
    for s in range(K):
        B = get_AXB(A=V.T, B=B_list[s] @ W)
        C = np.column_stack([VtBiW_list[i].ravel() for i in range(K)])
        D = np.hstack([Fq.Zeros((b1*b2, s*K))] + [-C]
                      + [Fq.Zeros((B.shape[0], (K-1-s)*K))] + [B])
        E[s*b1*b2:(s+1)*b1*b2,:] = D
    rank = np.linalg.matrix_rank(E)
    nullity = E.shape[1] - rank
    nullity_expect = m + (m+ell1-b1)*(m+ell1)
    if nullity != nullity_expect:
        return False # failure
    if not suppress_output: print("\nMinRank-like equation step done.")

    # remove random columns
    NS = E.null_space()
    L = Fq.Zeros((0,n+ell2))
    # 20 is arbitrary; this should succeed in the first iteration with high probability
    for _ in range(20):
        F, mu = draw_random_F_mu_from_solution_space(Fq=Fq, NS=NS, K=K, m=m, ell1=ell1)
        for s in range(K):
            sum_mu_Bi = Fq.Zeros(B_list[0].shape)
            for i in range(B_list.shape[0]): sum_mu_Bi += mu[s][i]*B_list[i]
            temp = V.T @ ((F @ B_list[s]) - sum_mu_Bi)
            L = np.vstack([L, temp]).row_space()
            if L.shape[0] == ell2:
                break
        if L.shape[0] == ell2:
            break
    if L.shape[0] != ell2:
        return False # failure
    Lperp = L.null_space().row_reduce().T
    # check if the obtained space spans the orthogonal complement of the last ell2 columns of Q
    if not (Lperp == W_true).all():
        return False # should not happen if we reached this point
    # the actual "remove random columns" step
    B_listp = Fq.Zeros((K, (m+ell1), n))
    for i in range(K):
        B_listp[i] = B_list[i] @ Lperp
    if not suppress_output: print("Removing random columns step done.")

    # remove random rows
    # again 20 is arbitrary; we expect we only need on iteration
    for _ in range(20):
        _, mu = draw_random_F_mu_from_solution_space(Fq=Fq, NS=NS, K=K, m=m, ell1=ell1)
        # Solve F'*Bs = [I F'] M sum_i (mu_{i,s}*B_i) for all s
        # Note H_s := M sum_i (mu_{i,s}*B_i) is known after we pick M
        # so this is equivalent to solving
        #     F'*Bs - [I F''] H_s = 0 for all s
        #  or F'*Bs - H1_s - F''*H2_s = 0 for all s
        # where we split H_s into two matrices with its first m rows and remaining rows respectively
        # Note H1_s is completely known. We make this system homogeneous by scaling it with an
        # unknown scalar lambda:
        #    F'*Bs - lambda*H1_s - F''*H2_s = 0 for all s
        # After solving this we can pick any solution with lambda != 0.
        E = Fq.Zeros((K*m*n, m*(m+ell1) + m*ell1 + 1))
        last = 0 # index of last filled row of E
        M = get_random_full_rank_matrix(Fq, m+ell1, m+ell1)
        for s in range(K):
            sum_mu_Bi = Fq.Zeros(B_listp[0].shape)
            for i in range(B_list.shape[0]): sum_mu_Bi += mu[s][i]*B_listp[i]
            H = M @ sum_mu_Bi # apply random invertible matrix
            H1 = H[:m,:] # first m rows of H
            H2 = H[m:,:] # remaining rows of H
            # B is such that B*vec(F') = vec(F'*B_s)
            B = get_AXB(Fq.Identity(m), B_listp[s])
            # C is such that C*vec(F'') = vec(F''*H2_s)
            C = get_AXB(Fq.Identity(m), H2)
            # D*[vec(F') vec(F'') lambda]^T = vec(F'*Bs - F''*H2_s - lambda*H1_s)
            D = np.hstack((B, -C, -H1.ravel().reshape(-1, 1))).row_space()
            E[last:last+D.shape[0],:] = D
            last = last + D.shape[0]
            rank = np.linalg.matrix_rank(E)
            nullity = E.shape[1] - rank
            # stop early if we already have a solution space of dimension 1
            if nullity == 1:
                break
        if nullity == 1:
            break
    if nullity != 1:
        return False # failure
    NS = E.null_space()
    assert NS.shape[0] == 1
    NS = NS[0] # so that it has shape (x,) and not (1,x)
    assert NS[-1] != 0 # lambda should not be zero
    Fp = NS[:m*(m+ell1)].reshape((m, m+ell1)) # obtain Fp
    Fp = Fp.row_reduce()
    if not (Fp.T == V_true).all():
        return False # should not happen if we reached this point
    # the actual "remove random rows" step
    B_listpp = Fq.Zeros((K, m, n))
    for i in range(K):
        B_listpp[i] = Fp @ B_listp[i]
    if not suppress_output: print("Removing random rows step done.")

    # Now we show there exist two matrices S1 and S2 such that
    #     B_listpp[i] = S1 * A_list[i] * S2
    # where recall A_list is the basis of the secret matrix code Cmat
    S1 = (Fp @ P)[:,:m]
    S2 = (Q @ Lperp)[:n,:]
    for i in range(K):
        diff = B_listpp[i] - (S1 @ A_list[i] @ S2)
        if np.any(diff):
            return False # failure
    if not suppress_output: print("Derived matrix codes is equivalent to secret matrix code.")
    return True


def draw_random_F_mu_from_solution_space(Fq, NS, K, m, ell1):
    """Helper function to draw a random solution (F, mu) obtained by solving the MinRank-like
    equation.
    """
    while True:
        r = Fq.Random(NS.shape[0]) @ NS
        mu = r[: K * K].reshape((K, K))  # mu_{s,i} = mu[s][i]
        F = r[K * K :].reshape(((m + ell1), (m + ell1)))
        # If mu is a diagonal matrix then the underlying T in F is actually a scalar matrix which is
        # not useful. (Checking if mu is not a scalar matrix would also have been sufficient.)
        if not is_diagonal(mu):
            return F, mu


def is_diagonal(M):
    """Return true if M is a diagonal matrix."""
    Fq = M.__class__
    MM = Fq.Zeros(M.shape)
    for i in range(M.shape[0]):
        MM[i, i] = M[i, i]
    if np.any(M - MM):
        return False
    return True
