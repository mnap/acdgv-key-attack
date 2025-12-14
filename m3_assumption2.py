from math import ceil

import numpy as np
import galois

from common import add_random_rows_columns
from common import get_matrix_code_expanded_using_power_basis_of_Fqm
from common import do_linear_comb_matrices
from common import get_gabidulin_generator_matrix
from common import get_random_full_rank_Fq_vector
from common import get_random_full_rank_matrix
from common import get_random_vector
from common import left_multiply_matrix_basis
from common import set_global_rng


def run(q, m, n, k, ell1, code_family, optimize, seed):
    """Tests Assumption 2 on the given parameters.
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
        raise ValueError("code_family should be either GABIDULIN or RANDOM.")

    # Get basis of Cmat by expanding the Fqm-linear code spanned by the generator matrix G_Cvec.
    # The matrix basis has a specific order (see its docstring) and the expansion is done using a
    # "power basis" of Fqm.
    A_list = get_matrix_code_expanded_using_power_basis_of_Fqm(G_Cvec) # shape (km, m, n)

    # The following constructs musi such that \sum_i musi[s][i]*A_list[i] = T*A_list[i] is the
    # matrix representation of multiplying b with the codeword in Cvec corresponding to the matrix
    # A_list[s]. This makes use of the fact that get_matrix_code_expanded_using_power_basis_of_Fqm
    # returns a specific ordered basis of the matrix code (see its docstring).
    # Note if dim Stab(Cvec) = m, then drawing a T uniformly that is not a scalar matrix and
    # that satisfies \sum_i musi[s][i]*A_list[i] = T*A_list[i] for all s
    # is equivalent to uniformly drawing a b \in F_{q^m} not in F_q.
    alpha = Fqm("x")
    while True:
        b = get_random_vector(Fqm, 1)[0]
        if b**q - b != 0:
            break # check b not in subfield
    musi = Fq.Zeros((K, K))
    for s1 in range(k):
        for s2 in range (m):
            # (alpha^i).vector() equals [0 0 0 ... 1 ... 0] where 1 is at position (m-i).
            # For example, alpha.vector() = [0 0 0 ... 1 0].
            # [::-1] reverses the vector.
            musi[s1*m+s2, s1*m:(s1+1)*m] = (b*(alpha**s2)).vector()[::-1]

    # It is not hard to see that Assumption 2 does not depend on using a random basis of F_{q^m} or
    # a random basis of A_list. Thus we skip randomizing these bases when optimize=True.
    if not optimize:
        # randomize Fq-linear basis of Fqm used to expand Cvec to Cmat
        A_list = left_multiply_matrix_basis(get_random_full_rank_matrix(Fq, m, m), A_list)
        # draw a new random basis of Cmat
        X = get_random_full_rank_matrix(Fq, K, K)
        A_list = do_linear_comb_matrices(A_list, X)
        musi = X @ musi @ np.linalg.inv(X)
    AR_list = add_random_rows_columns(A_list, ell1, 0)  # shape (km, m+ell1, n)
    AR_list_RHS = do_linear_comb_matrices(AR_list, musi)

    # Note Assumption 2 is true iff the system
    # [J1 J2] [As // Rs] = [K1 K2]\sum_i \mu_{is}*[Ai // Ri] for all s
    # for unknowns [J1 J2] and [K1 K2] only has solutions J2 = K2 = 0 (so that J1 = K1*T).
    # (Here // denotes concatenating vertically, i.e., one below the other.)
    # This is true iff the solution space at most dimension m^2 since the solutions J2 = K2 = 0 and
    # J1 = K1*T are provably in the solution space and are also of dimension m^2.
    # Now the system is equivalent to
    # [J1 K1 // J2 K2] [As // Rs // AAs // RRs] = 0 for all s
    # where we let [AAs // RRs] := \sum_i \mu_{is}*[Ai // Ri] (in our code this is AR_list_RHS[s])
    # Note that the solution space of the above is m times the solution space of the below system
    #   x [As // Rs // AAs // RRs] = 0 for all s
    # where the unknown x is just a row vector. Thus we need to check if the dimension of the left
    # null space of the matrix obtained by horizontally stacking [As // Rs // AAs // RRs] for s = 1,
    # ..., km equals m.
    # Finally, since we only need to check that the dimension is at most m, we only need to check if
    # its rank is at least (rows in the big horizontally stacked matrix) - m. Thus, the optimized
    # version checks only a subset of this matrix's columns and if it has sufficient rank, it can
    # stop early.
    if not optimize:
        E = Fq.Zeros((2*(m+ell1), n*K))
        expected_left_nullity = m
        for s in range(K):
            C = np.vstack([AR_list[s], AR_list_RHS[s]])
            E[:, s*n:(s+1)*n] = C
        rank = np.linalg.matrix_rank(E)
        left_nullity = E.shape[0] - rank
        return left_nullity == expected_left_nullity
    else:
        E = Fq.Zeros((2*(m+ell1), n*ceil((2*(m+ell1) - m)/n)))
        expected_left_nullity = m
        for s in range(K):
            C = np.vstack([AR_list[s], AR_list_RHS[s]])
            E[:, s*n:(s+1)*n] = C
            if (s+1)*n >= 2*(m+ell1) - m:
                rank = np.linalg.matrix_rank(E)
                if rank <= E.shape[0] - expected_left_nullity:
                    break
                # else append zero columns to E so we can do one more iteration
                E = np.vstack((E, Fq.Zeros((E.shape[0], n))))
        left_nullity = E.shape[0] - rank
        return left_nullity == expected_left_nullity
