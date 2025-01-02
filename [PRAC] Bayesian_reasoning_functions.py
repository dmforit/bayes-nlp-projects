import numpy as np
from scipy.stats import binom, poisson
from scipy.signal import convolve


def pa(params, model):
    amin = params['amin']
    amax = params['amax']
    val = np.arange(amin, amax + 1) 
    prob = np.array([1 / (amax - amin + 1) for _ in val])
    return prob, val


def pa_single(params, model):
    amin = params['amin']
    amax = params['amax']
    return 1 / (amax - amin + 1)


def pb_single(params, model):
    bmin = params['bmin']
    bmax = params['bmax']
    return 1 / (bmax - bmin + 1)
    

def pb(params, model):
    bmin = params['bmin']
    bmax = params['bmax']
    val = np.arange(bmin, bmax + 1) 
    prob = np.array([1 / (bmax - bmin + 1) for _ in val])
    return prob, val


def pc_ab(a, b, params, model):
    val = np.arange(params['amax'] + params['bmax'] + 1)
    prob = np.zeros((val.shape[0], a.shape[0], b.shape[0]))

    if model == 3:
        a_vals = np.arange(params['amax'] + 1).reshape(-1, 1)
        b_vals = np.arange(params['bmax'] + 1).reshape(-1, 1)

        p_first = binom.pmf(a_vals, a.reshape(1, -1), params['p1'])
        p_second = binom.pmf(b_vals, b.reshape(1, -1), params['p2'])

        p_c = convolve(p_first[:, :, np.newaxis], 
                       p_second[:, np.newaxis, :], 
                       mode='full')
        
        prob[:p_c.shape[0], :, :] = p_c
        prob[prob < 0] = 0
    elif model == 4:
        lambdas = np.outer(a, params['p1']) + np.outer(b, params['p2']).T
        prob = poisson.pmf(val[:, np.newaxis, np.newaxis], lambdas[np.newaxis, :, :])

    return prob, val
    

def pc(params, model):
    prob_c_ab, val  = pc_ab(np.arange(params['amin'], params['amax'] + 1), 
                      np.arange(params['bmin'], params['bmax'] + 1), 
                      params, model)
    return prob_c_ab.sum(axis=(1, 2)) * pa_single(params, model) * pb_single(params, model), val


def pd_c(c, params, model):
    val = np.arange(2 * (params['amax'] + params['bmax']) + 1)
    prob = np.zeros((val.shape[0], c.shape[0]))
    for c_i, c_val in enumerate(c):
        if (c_val < 0) or (c_val > params['amax'] + params['bmax']):
            continue
        pr = binom.pmf(np.arange(c_val + 1), c_val, params['p3'])
        prob[c_i:2*c_i + 1, c_i] = pr
    return prob, val


def pd(params, model):
    prob_c, val_c = pc(params, model)
    prob_d_c, val = pd_c(val_c, params, model)
    return prob_d_c @ prob_c, val


def pd_ab(a, b, params, model):
    prob_c_ab, val_c = pc_ab(a, b, params, model)
    prob_d_c, val = pd_c(val_c, params, model)
    return np.tensordot(prob_d_c, prob_c_ab, axes=(1, 0)), val


def generate(N, a, b, params, model):
    prob_d_ab, val = pd_ab(a, b, params, model)
    samples = np.zeros((N, a.shape[0], b.shape[0]), dtype=int)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            probs = prob_d_ab[:, i, j]
            samples[:, i, j] = np.random.choice(val, size=N, p=probs)
    return samples

    
def pb_d(d, params, model):
    p_d_ab, _ = pd_ab(np.arange(params['amin'], params['amax'] + 1), 
                      np.arange(params['bmin'], params['bmax'] + 1),
                      params, model)
    
    val = np.arange(params['bmin'], params['bmax'] + 1)
    prob = np.zeros((val.shape[0], d.shape[0]))

    for k_d in range(d.shape[0]):
        prod_pd_given_ab = p_d_ab[d[k_d]].prod(axis=0) 
        prob[:, k_d] = np.sum(prod_pd_given_ab, axis=0) / np.sum(prod_pd_given_ab)
    return prob, val


def pb_ad(a, d, params, model):
    p_d_ab, _ = pd_ab(np.arange(params['amin'], params['amax'] + 1), 
                      np.arange(params['bmin'], params['bmax'] + 1),
                      params, model)
    val = np.arange(params['bmin'], params['bmax'] + 1)
    prob = np.zeros((val.shape[0], a.shape[0], d.shape[0]))

    for k_d in range(d.shape[0]):
        prod_pd_given_ab = p_d_ab[np.ix_(d[k_d], a - params['amin'], val - params['bmin'])].prod(axis=0)
        prob[:, :, k_d] = prod_pd_given_ab.T / np.sum(prod_pd_given_ab.T, axis=0)
    
    return prob, val
