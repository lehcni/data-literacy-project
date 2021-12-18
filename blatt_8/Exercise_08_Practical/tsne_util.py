import numpy as np



def perp_fn(i, beta_i, dists, perplexity_function):
    """
    Function that takes an index i, beta_i, and all pairwise distances of X
    and return the perplexity of p_{j|i} (Eq. 1 in the paper).
    """
    exp_dists = np.exp(-dists[i] / beta_i)
    exp_dists[i] = 0
    p_j_given_i = exp_dists / exp_dists.sum()

    perp_i = perplexity_function(p_j_given_i)

    return perp_i


def binary_search(perp, dists, perplexity_function):
    """
    Let beta_i := 2 \sigma_i^2. This function computes (beta_i) that achieve
    the desired perplexity.
    
    Params:                 
        perp: Desired perplexity value.
        
        dists: Pairwise squared Euclidean distances, stored in an (n x n)-matrix
        
        perplexity_function: A function that return the perplexity number given a probability vector
        
    Returns:
        betas: (n,) array of beta_i's 
    """    
    n = len(dists)
    betas = []
    
    for i in range(n):
        # Binary search
        min_beta, max_beta = 1e-10, 1e10

        for _ in range(1000):
            mid_beta = (min_beta + max_beta) / 2
            p_mid = perp_fn(i, mid_beta, dists, perplexity_function)

            if p_mid >= perp:
                max_beta = mid_beta
            else:
                min_beta = mid_beta

            # Close enough, use the current mid value
            if abs(p_mid - perp) < 1e-3:
                break
                
        betas.append(mid_beta)

    return np.array(betas)


def test_get_dists(get_dists, X):
    n = len(X)
    return get_dists(X).shape == (n, n)


def test_get_p_j_given_i(get_p_j_given_i, X):
    n = len(X)
    perp = 30
    P = get_p_j_given_i(X, perp)
    test1 = np.allclose(P.sum(1), np.ones(n))
    test2 = np.allclose(np.diag(P), np.zeros(n))
    return test1 and test2


def test_get_get_P(get_P, X):
    n = len(X)
    perp = 30
    P = get_P(X, perp)
    test1 = np.allclose(P.sum(), 1)
    test2 = np.allclose(np.diag(P), np.zeros(n))
    return test1 and test2


def test_get_Q(get_dists, get_Q):
    n = 100
    n_proj_dim = 2
    Y = np.random.randn(n, n_proj_dim) * 10**(-2)
    dists_y = get_dists(Y)
    Q = get_Q(dists_y)
    test1 = np.allclose(Q.sum(), 1)
    test2 = np.allclose(np.diag(Q), np.zeros(n))
    return test1 and test2


def test_get_grad(get_grad, get_P, get_dists, get_Q, X):
    n = len(X)
    n_proj_dim = 2
    perp = 30
    Y = np.random.randn(n, n_proj_dim) * 10**(-2)
    P = get_P(X, 30)
    dists_Y = get_dists(Y)
    Q = get_Q(dists_Y)
    return get_grad(Y, P, Q, dists_Y).shape == (n, n_proj_dim)
