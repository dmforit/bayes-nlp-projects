import numpy as np
from scipy.stats import norm
from scipy.signal import fftconvolve
from scipy.special import softmax


EPS = 1e-16


def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh, dw, k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """
    H, W, _ = X.shape
    h, w = F.shape
    part_XB = (X - B[..., None]) ** 2
    part_F = (fftconvolve(X * X, np.ones((h, w, 1)), mode="valid") -
              2 * fftconvolve(X, F[::-1, ::-1, None], mode="valid") +
              (F * F).sum())
    part_B = (np.sum(part_XB, axis=(0, 1)) -
              fftconvolve(part_XB, np.ones((h, w, 1)), mode="valid"))
    ll = - H * W * (np.log((2 * np.pi) ** 0.5) + np.log(s + EPS)) - (part_F + part_B) / (2 * (s ** 2))

    return ll


def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    ll = calculate_log_probability(X, F, B, s)
    indices = (q[0], q[1], np.arange(q.shape[1]))

    if use_MAP:
        A = np.where(A <= 0, EPS, A)
        return np.sum((ll + np.log(A)[..., None])[indices])
    else:
        return np.sum(q * (ll + np.log(A)[..., None] - np.log(q + EPS)))


def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """
    ll = calculate_log_probability(X, F, B, s) + np.log(A + EPS)[..., None]
    q = softmax(ll - np.max(ll, axis=(0, 1), keepdims=True), axis=(0, 1))
    return q if not use_MAP else np.array(np.unravel_index(np.argmax(q.reshape(-1, q.shape[-1]), axis=0), shape=q.shape[:-1]))


def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    H, W, K = X.shape

    if use_MAP:
        q_map = np.zeros((H - h + 1, W - w + 1, K))
        q_map[q[0], q[1], np.arange(q.shape[1])] = 1
    else:
        q_map = q

    # A estimation

    A = np.sum(q_map, axis=2) / K

    # F estimation

    F = np.sum(fftconvolve(X, q_map[::-1, ::-1], axes=(0, 1), mode="valid"), axis=2) / K

    # B estimation

    weight = np.sum(q_map, axis=(0, 1)) - fftconvolve(q_map, np.ones((h, w, 1)), mode="full")
    B = np.sum(weight * X, axis=-1) / np.sum(weight, axis=-1)

    # s estimation

    part_XB = (X - B[..., None]) ** 2
    part_F = (fftconvolve(X * X, np.ones((h, w, 1)), mode="valid") -
              2 * fftconvolve(X, F[::-1, ::-1, None], mode="valid") +
               (F * F).sum())
    part_B = (np.sum(part_XB, axis=(0, 1)) -
              fftconvolve(part_XB, np.ones((h, w, 1)), mode="valid"))

    s2 = (q_map * (part_F + part_B)).sum() / (H * W * K)
    s2 = s2 if s2 >= 0 else EPS

    return F, B, np.sqrt(s2), A


def initialize_parameters(X, h, w):
    """
    Initialize F, B, s, A if not provided.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.

    Returns
    -------
    F, B, s, A : initialized parameters
    """
    H, W, _ = X.shape

    rand_x = np.random.randint(0, H - h + 1)
    rand_y = np.random.randint(0, W - w + 1)
    F = X[rand_x:rand_x + h, rand_y:rand_y + w, 0]
    B = np.mean(X, axis=2)
    s = np.std(X)
    A = np.ones((H - h + 1, W - w + 1)) / ((H - h + 1) * (W - w + 1))

    return F, B, s, A


def initialize_parameters_dirichlet_normal(X, h, w):
    """
    Initialize F, B, s, A using Dirichlet or Normal distributions.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.

    Returns
    -------
    F, B, s, A : initialized parameters
    """
    H, W, K = X.shape

    F = np.random.normal(loc=0.5, scale=0.1, size=(h, w))
    F = np.clip(F, 0, 1)

    B = np.random.normal(loc=0.5, scale=0.1, size=(H, W))
    B = np.clip(B, 0, 1)

    s = np.abs(np.random.normal(loc=0.1, scale=0.05))

    flat_A = np.random.dirichlet(alpha=np.ones((H - h + 1) * (W - w + 1)))
    A = flat_A.reshape((H - h + 1, W - w + 1))

    return F, B, s, A


def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters,)
        L(q,F,B,s,A) after each EM iteration (1 iteration = 1 e-step + 1 m-step);
        number_of_iters is actual number of iterations that was done.
    """
    if any(param is None for param in [F, B, s, A]):
        F, B, s, A = initialize_parameters(X, h, w)

    elbo = [-1]
    for _ in range(max_iter):
        q = run_e_step(X, F, B, s, A, use_MAP)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP)
        elbo.append(calculate_lower_bound(X, F, B, s, A, q, use_MAP))
        if abs(elbo[-1] - elbo[-2]) < tolerance:
            break
    return F, B, s, A, elbo[1:]


def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """
    best_l = -1
    best_params_values = []
    for _ in range(n_restarts):
        F, B, s, A = initialize_parameters_dirichlet_normal(X, h, w)
        res = run_EM(X, h, w, F=F, B=B, s=s, A=A,
                     tolerance=tolerance, max_iter=max_iter, use_MAP=use_MAP)
        if res[-1][-1] > best_l:
            best_params_values = res[:-1] + res[-1][-1]
    return tuple(best_params_values)
