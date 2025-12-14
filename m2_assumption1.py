from math import ceil

import numpy as np
import galois

from common import add_random_rows_columns
from common import get_matrix_code_expanded_using_power_basis_of_Fqm
from common import get_gabidulin_generator_matrix
from common import get_random_full_rank_Fq_vector
from common import get_random_full_rank_matrix
from common import get_AXB
from common import left_multiply_matrix_basis
from common import set_global_rng


def run(q, m, n, k, ell1, b1, b2, code_family, optimize, seed):
    """Tests Assumption 1 on the given parameters.
    If code_family = "RANDOM" (resp. "GABIDULIN") then a random F_{q^m}-linear (Gabidulin) code is
    used. If optimize=True, then certain tricks are used to run this test faster (suitable for
    larger parameters), but the downside is that the code is more complex to verify. The seed
    argument is for reproducibility.
    """
    set_global_rng(seed)
    K = k*m
    Fqm = galois.GF(q, m)
    Fq = Fqm.prime_subfield
    if code_family == "GABIDULIN":
        g = get_random_full_rank_Fq_vector(Fqm, n)
        G_Cvec = get_gabidulin_generator_matrix(g, k)
    elif code_family == "RANDOM":
        G_Cvec = get_random_full_rank_matrix(Fqm, k, n)
    else:
        raise ValueError("code_family should either be GABIDULIN or RANDOM.")

    # One can show that Assumption 1 does not depend on the which basis of Fqm over Fq is used to
    # obtain the matrix code Cmat, nor does it depend on which km-length basis we pick for Cmat.
    # Thus, when optimize=True, we skip randomizing these things.
    if not optimize:
        A_list = get_matrix_code_expanded_using_power_basis_of_Fqm(G_Cvec, random_basis=True) # shape (km, m, n)
        # use a new random basis of Fqm over Fq
        A_list = left_multiply_matrix_basis(get_random_full_rank_matrix(Fq, m, m), A_list)
    else:
        # Interestingly, if we use Gabidulin codes and do not pick a random basis of the
        # corresponding matrix code then helper_asm1_find_rank_optimized becomes quite inefficient.
        # Hence, we randomize the basis even when optimize=True.
        if code_family == "GABIDULIN":
            A_list = get_matrix_code_expanded_using_power_basis_of_Fqm(G_Cvec, random_basis=True) # shape (km, m, n)

    # AR_list[i] = [A_i // R_i] \in \Fq^{(m+ell1) x m}
    AR_list = add_random_rows_columns(A_list, ell1, 0)  # shape (km, m+ell1, n)
    U1 = get_random_full_rank_matrix(Fq, b1, m)
    U2 = get_random_full_rank_matrix(Fq, n, b2)

    # AR_U2_list[i] = [A_i // R_i]*U2
    # AR_list[i] = [A_i // R_i]*U2
    AR_U2_list = Fq.Zeros((K, m+ell1, b2))
    U1_A_U2_list = Fq.Zeros((K, b1, b2))
    for i in range(K): AR_U2_list[i] = AR_list[i] @ U2
    for i in range(K): U1_A_U2_list[i] = U1 @ A_list[i] @ U2

    if not optimize:
        ret = helper_asm1_find_rank(U1, U1_A_U2_list, AR_U2_list, m, ell1, K, b1, b2)
    else:
        ret = helper_asm1_find_rank_optimized(U1_A_U2_list, AR_U2_list, m, ell1, K, b1, b2)
    return ret


def helper_asm1_find_rank(U1, U1_A_U2_list, AR_U2_list, m, ell1, K, b1, b2):
    r"""Solves the system
    U1*J*[A_i // R_s]*U2 - \sum_i mu_{i,s}*U1*A_i*U2 = 0 for s = 1, ..., km
    for unknowns J \in \Fq^{m \times (m+ell1)} and mu_{i,s} by vectorizing it using the Kronecker
    product (see definition of get_AXB). Here U1 has size b1 x m, [A_i // R_s] has size (m+ell1) x
    n, U2 has size n x b2 and A_i has size m x n. Returns True if the dimension of the solution
    space matches Assumption 1.

    In the below code, matrices B, C, D in iteration s are such that:
        B*vec(J) = vec(U1*J*[A_s // R_s]*U2)
        C*[mu_{s,1}, ..., mu_{s,km}]^T = vec(\sum_i mu_{s,i} U1*A_i*U2)
        D = [0...0 -C 0...0 B] where -C is at position s
        hence D*[{mu_{1,1}, ...,  mu_{km,km}, vec(J)^T]^T
        = vec(U1*J*[A_s // R_s] - \sum_i mu_{s,i} U1*A_i*U2)
    The matrix E = [D_1 // D_2 // ... // D_{km}] is the vertical stacking of all the D matrices
    computed above.
    """
    Fq = U1.__class__
    E = Fq.Zeros((K*b1*b2, K*K + m*(m+ell1)))
    for s in range(K):
        B = get_AXB(U1, AR_U2_list[s])
        C = np.column_stack([U1_A_U2_list[i].ravel() for i in range(K)])
        D = np.hstack([Fq.Zeros((b1*b2, s*K))] + [-C] + [Fq.Zeros((b1*b2, (K-1-s)*K))] + [B])
        E[s*b1*b2:(s+1)*b1*b2,:] = D
    rank = np.linalg.matrix_rank(E)
    nullity = E.shape[1] - rank
    expected_nullity = m + (m-b1)*(m+ell1)
    return nullity == expected_nullity

def helper_asm1_find_rank_optimized(U1_A_U2_list, AR_U2_list, m, ell1, K, b1, b2):
    r"""Same behaviour as the non-optimized version but uses some tricks to speed things up:
    1. Considers U1*J as a new unknown instead of J.
    2. The vectorized system of assumption 1 looks like (see also Lemma 6 in Appendix E of paper):
               [L          S_1   ]
       Z =     [  L        S_2   ]
               [    ...
               [        L  S_{km}]
       where L = [vec(U1*A_1*U2) ... vec(U1*A_{km}*U2)] \in Fq^{b1*b2 x km}
       and S_i = kronecker_product(I_b1, [A_i // R_i]*U2) \in Fq^{b1*b2 x (m+ell1)*b1}
    Thus to find the rank of Z, one first row reduces L in each (block) row of Z.
    If the rank of L is r, then the rank of Z is kmr + r' where r' is the rank
    of the matrix obtained by restricting Z to only those rows where ref(L) is zero. Note that
    these rows only have non-zero entries in the last columns corresponding to the positions of
    S_i's. Further, we can obtain these rows by multiplying a basis of the left null space
    of L with S_i. Stacking the results for all i vertically we obtain the (tall) matrix E (of size
    wkm x b1*(m+ell1) where w is the dimension of the left null space of L). We then find the rank
    of E.
    3. The computation of [matrix whose rows span the left null space of L]*S_i can be done
    faster by using the fact that S_i = kronecker_product(I_b1, [A_i // R_i]*U2).
    Indeed, the lines
        for t in range(b1):
          E[s*w:(s+1)*w, t*(m+ell1):(t+1)*(m+ell1)] = L[:,t*b2:(t+1)*b2] @ Bs
    are equivalent to
        E[s*w:(s+1)*w,:] = L @ get_AXB(Fq.Identity(b1), AR_U2_list[s])
    except the former takes less time.
    4. Assumption 1 is true iff Z has nullity m (it is instead m + (m-b1)*(m+ell1) if we treat J as
    the unknown and not U1*J as we are doing here). In fact, we know that it provably has nullity at
    least m (see paper), so we only need to check if nullity(Z) is at most m. This also means
    nullity(E) is at most m. This means rank(E) is at least b1(m+ell1) - m. Thus if we determine
    that only a subset of rows of E has sufficient rank then we can stopy early.
    """
    K, b1, b2 = U1_A_U2_list.shape
    Fq = U1_A_U2_list.__class__
    L = np.column_stack([U1_A_U2_list[i].ravel() for i in range(K)]).left_null_space()
    w = L.shape[0] # L has shape (w, b1*b2)
    E = Fq.Zeros((w*ceil((b1*(m+ell1))/w), b1*(m+ell1)))
    expected_nullity = m
    for s in range(K):
        Bs = AR_U2_list[s].T # shape b2 x (m+ell1)
        for t in range(b1):
            E[s*w:(s+1)*w, t*(m+ell1):(t+1)*(m+ell1)] = L[:,t*b2:(t+1)*b2] @ Bs
        if (s+1)*w >= b1*(m+ell1):
            rank = np.linalg.matrix_rank(E)
            if rank >= E.shape[1] - expected_nullity:
                break
            # else append w zero rows to E so we can do one more iteration
            E = np.vstack((E, Fq.Zeros((w, E.shape[1]))))
    nullity = E.shape[1] - rank
    return nullity == expected_nullity
