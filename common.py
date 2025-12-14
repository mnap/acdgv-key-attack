from math import ceil

import numpy as np

global_rng = None


def set_global_rng(seed):
    global global_rng
    global_rng = np.random.default_rng(seed)


def get_random_vector(Fqm, n):
    """Return random length-n vector over Fqm."""
    return Fqm.Random(n, seed=global_rng)


def get_random_full_rank_matrix(Fqm, a, b):
    """Return a random a x b matrix of full rank over Fqm.

    Uses rejection sampling, so is quite slow.
    """
    while True:
        matrix = Fqm.Random((a, b), seed=global_rng)
        if np.linalg.matrix_rank(matrix) == min(a, b):
            return matrix


def get_random_full_rank_Fq_vector(Fqm, n):
    """Return random length-n vector over Fqm with entries linearly independent over its prime
    field.
    """
    if Fqm.degree < n:
        raise ValueError("The field extension degree must be at least the number of elements.")
    matrix = get_random_full_rank_matrix(Fqm.prime_subfield, n, Fqm.degree)  # shape (n, m)
    return Fqm.Vector(matrix)


def get_AXB(A, B):
    """Return matrix C such that C*vec(X) = vec(A*X*B) where vec(X) is the vector obtained by
    stacking the rows of X into a column vector (equivalent to np.ravel(X, order='C')).

    If matrices A and B have sizes a1 x a2 and b1 x b2 respectively, then this returns a matrix of
    size a1*b2 x a2*b1.
    """
    return np.kron(A, B.T)


def get_matrix_code_expanded_using_power_basis_of_Fqm(G, random_basis=False):
    r"""Return a list of km matrices that forms the basis of the Fq-linear matrix code obtained by
    expanding the Fqm-linear code rowspan(G) using the Fq-basis (1, alpha, alpha^2, ...,
    alpha^{m-1}) of Fqm.

    The k x n matrix G is the generator matrix of an Fqm-linear code. The element "alpha" is the
    residue class of x in Fqm \cong Fq[x]/<f(x)> where f(x) is the irreducible polynomial (=
    Fqm.irreducible_poly) used to construct Fqm.

    The return value is a list of km matrices with each matrix of size m x n. If random_basis=False,
    then, if g_i is the ith row of G, the list of matrices obtained is: ext(g_1), ext(alpha*g_1),
    ..., alpha^{m-1}*g_1, ext(g_2), ext(alpha*g_2), ..., alpha^{m-1}*g_k where ext(.) returns the m
    x n matrix over Fq obtained by expanding its input n-length vector using the basis (1, alpha,
    alpha^2, ..., alpha^{m-1}). Note this is not a random basis of the corresponding matrix code. To
    get a random basis, do random_basis=True.
    """
    Fqm = G.__class__
    Fq = Fqm.prime_subfield
    alpha = Fqm("x")
    k, n = G.shape
    m = Fqm.degree
    alpha_powers = alpha ** np.arange(m)  # shape (m,)
    # note this groups the rows g_i, alpha^1*g_i, ... alpha^(m-1)*g_i next to each other for each
    # row g_i of G
    G_tall = (G[:, None, :] * alpha_powers[None, :, None]).reshape(-1, G.shape[1]) # shape (m*k, n)
    if random_basis:
        X = Fqm(get_random_full_rank_matrix(Fq, k*m, k*m))
        G_tall = X @ G_tall
    # if random_basis=False then all the alpha^i * g_j rows for any fixed j are next to each other
    matrices = G_tall.vector()  # shape (m*k, n, m)
    matrices = matrices.transpose(0, 2, 1)  # shape (m*k, m, n)
    return matrices


def get_gabidulin_generator_matrix(g_vec, k):
    """Return the (k × n) generator matrix [g_vec // g_vec^q // ...// g_vec^{q^{k-1}}] of a
    Gabidulin code where g_vec is an n-length vector over Fqm. Here // denotes vertical stacking.
    """
    n = len(g_vec)
    assert k <= n
    Fqm = g_vec.__class__
    q = Fqm.characteristic
    G = Fqm.Zeros((k, n))
    powers = q ** np.arange(k)[:, None]  # shape (k, 1) for broadcasting
    G = g_vec**powers
    return G


def add_random_rows_columns(A_list, ell1, ell2):
    """Adds ell1 random rows and ell2 random columns to each matrix in A_list.

    If A_list has shape (K, m, n) then this returns a shape (K, m+ell1, n+ell2) array.
    Note the random rows and columns are appended to the matrix and not "mixed in" by left/right
    multiplication with invertible matrices.
    """
    K, m, n = A_list.shape
    Fq = A_list.__class__
    B_list = Fq.Zeros((K, m + ell1, n + ell2))
    for i in range(len(B_list)):
        R = Fq.Random((m, ell2), seed=global_rng)
        R_prime_12 = Fq.Random((ell1, n + ell2), seed=global_rng)
        temp = np.concatenate((A_list[i], R), axis=1)
        B_list[i] = np.concatenate((temp, R_prime_12), axis=0)
    return B_list


def left_multiply_matrix_basis(Y, A_list):
    """Return list of matrices B_list where the ith matrix B_list[i] = Y*A_list[i].

    The input A_list is a list of matrices of shape (K, m, n) and Y is a matrix of shape (m, n).
    """
    Fq = Y.__class__
    q = Fq.order
    if not is_prime(q):
        raise ValueError("q must be prime.")
    # the galois library only supports @ with 2-D arrays so we switch to numpy here
    # indeed this was the fastest among the alternatives we tried to do the multiplication
    Y_np = Y.view(np.ndarray)
    A_list_np = A_list.view(np.ndarray)
    A_list = ((Y_np @ A_list_np) % q).view(Fq)
    return A_list


def do_linear_comb_matrices(M_list, X):
    """Return a list of K' matrices where the jth matrix equals sum_i=1^K (X_{j,i} *
    M_list[i]).

    X is a matrix of size K' x K and M_list has shape (K, m, n) which represents K matrices each of
    size m x n. The output is a list of K' matrices where the jth matrix is a linear combination of
    the input M_list with coefficients from the jth row of X.
    """
    Fq = M_list.__class__
    q = Fq.order
    if not is_prime(q):
        raise ValueError("q must be prime.")
    Kp, m, n = X.shape[0], M_list.shape[1], M_list.shape[2]
    result = Fq.Zeros((Kp, m, n))
    # switch to numpy because the np.einsum is faster than the alternatives we tried
    Xv = X.view(np.ndarray)
    M_listv = M_list.view(np.ndarray)
    resultv = np.einsum('ij,jkl->ikl', Xv, M_listv)
    result = (resultv % Fq.order).view(Fq)
    return result


def get_b1_b2(k, m, n, ell1, ell2):
    """Pick the interal parameters (b1, b2) as required by the new key recovery algorithm."""
    min_cost = float("inf")
    best_b1, best_b2 = None, None
    for b1 in range(1, m + 1):
        min_b2 = ceil((k * m) / b1 + (m + ell1) / (k * m))
        for b2 in range(min_b2, n + 1):
            if b1 * (b2 - (m + ell1) / (k * m)) >= k * m:
                cost = b1 * ell1 + b2 * ell2
                if cost < min_cost:
                    min_cost = cost
                    best_b1, best_b2 = b1, b2
    if min_cost == float("inf"):
        raise ValueError("b1, b2 could not be computed correctly.")
    return best_b1, best_b2


def is_prime(n):
    if n < 2:
        return False
    else:
        for i in range(2, int(n**0.5)):
            if n % i == 0:
                return False
    return True
