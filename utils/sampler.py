from logging import error
import numpy as np
import torch
from torch.distributions import Dirichlet

def BetaSampler(bs, f, is_2d, p=None, beta_alpha=1): 

    shp = (bs, 1) if is_2d else (bs, 1, 1, 1)
    if p is None:
        alphas = []
        for i in range(bs):
            alpha = np.random.beta(beta_alpha, beta_alpha)    
            alphas.append(alpha)
    else:
        alphas = [p]*bs
    alphas = np.asarray(alphas).reshape(shp)
    alphas = torch.from_numpy(alphas).float()
    
    return alphas    

def UniformSampler(bs, f, is_2d, p=None): 
    shp = (bs, 1) if is_2d else (bs, 1, 1, 1)
    if p is None:
        alphas = []
        for i in range(bs):
            alpha = np.random.uniform(0, 1)
            alphas.append(alpha)
    else:
        alphas = [p]*bs
    alphas = np.asarray(alphas).reshape(shp)
    alphas = torch.from_numpy(alphas).float()
    
    return alphas

def UniformSampler2(bs, f, is_2d, p=None):
    """Mixup2 sampling function
    :param bs: batch size
    :param f: number of features / channels
    :param is_2d: should sampled alpha be 2D, instead of 4D?
    :param p: Bernoulli parameter `p`. If this is None, then we simply sample p ~ U(0,1).       
    :returns: an alpha of shape (bs, f) if `is_2d`, otherwise (bs, f, 1, 1).
    :rtype:
    """
    print('flag:UniformSampler2 ing')
    shp = (bs, f) if is_2d else (bs, f, 1, 1)
    if p is None:
        alphas = np.random.uniform(0, 1, size=shp)
    else:
        alphas = np.zeros(shp)+p
    alphas = torch.from_numpy(alphas).float()
    
    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        alphas = alphas.cuda()
    return alphas

def BernoulliSampler(bs, f, is_2d, p=None):

    shp = (bs, f) if is_2d else (bs, f, 1, 1)

    if p is None:
        alphas = torch.bernoulli(torch.rand(shp)).float()

    else:
        rnd_state = np.random.RandomState(0)
        rnd_idxs = np.arange(0, f)
        rnd_state.shuffle(rnd_idxs)
        rnd_idxs = torch.from_numpy(rnd_idxs)
        how_many = int(p*f)
        alphas = torch.zeros(shp).float()
        if how_many > 0:
            rnd_idxs = rnd_idxs[0:how_many]
            alphas[:, rnd_idxs] += 1.

    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        alphas = alphas.cuda()
    return alphas

def BernoulliSampler2(bs, f, is_2d, p=None):
    shp = (bs, f) if is_2d else (bs, f, 1, 1)
    if p is None:
        this_p = torch.rand(1).item()
        alphas = torch.bernoulli(torch.zeros(shp)+this_p).float()
    else:
        rnd_state = np.random.RandomState(0)
        rnd_idxs = np.arange(0, f)
        rnd_state.shuffle(rnd_idxs)
        rnd_idxs = torch.from_numpy(rnd_idxs)
        how_many = int(p*f)
        alphas = torch.zeros(shp).float()
        if how_many > 0:
            rnd_idxs = rnd_idxs[0:how_many]
            alphas[:, rnd_idxs] += 1.

    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        alphas = alphas.cuda()
    return alphas

def DirichletSampler(bs, f, is_2d, dirichlet_gama=9.0):
    with torch.no_grad():
        dirichlet = Dirichlet(torch.FloatTensor([dirichlet_gama, dirichlet_gama, dirichlet_gama]))
        alpha = dirichlet.sample((bs,))
        if not is_2d:
            alpha = alpha.reshape(-1, alpha.size(1), 1, 1)                              

    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        alpha = alpha.cuda()
    return alpha  
    
def BernoulliSampler3(bs, f, is_2d):          
    if is_2d:
        alpha = np.zeros((bs, 3, f)).astype(np.float32)    
    else:
        alpha = np.zeros((bs, 3, f, 1, 1)).astype(np.float32)
    for b in range(bs):
        for j in range(f):
            alpha[b, np.random.randint(0,3), j] = 1.
    alpha = torch.from_numpy(alpha).float()    

    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        alpha = alpha.cuda()
    return alpha  