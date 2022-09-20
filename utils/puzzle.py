import torch
import numpy as np
import gco
import torch.nn.functional 

def transport_image(img, plan, batch_size, block_num, block_size):
    '''apply transport plan to images'''
    input_patch = img.reshape([batch_size, 3, block_num, block_size,
                               block_num * block_size]).transpose(-2, -1)
    input_patch = input_patch.reshape([batch_size, 3, block_num, block_num, block_size,
                                       block_size]).transpose(-2, -1)
    input_patch = input_patch.reshape([batch_size, 3, block_num**2, block_size,
                                       block_size]).permute(0, 1, 3, 4, 2).unsqueeze(-1)

    input_transport = plan.transpose(
        -2, -1).unsqueeze(1).unsqueeze(1).unsqueeze(1).matmul(input_patch).squeeze(-1).permute(
            0, 1, 4, 2, 3)
    input_transport = input_transport.reshape(
        [batch_size, 3, block_num, block_num, block_size, block_size])
    input_transport = input_transport.transpose(-2, -1).reshape(
        [batch_size, 3, block_num, block_num * block_size, block_size])
    input_transport = input_transport.transpose(-2, -1).reshape(
        [batch_size, 3, block_num * block_size, block_num * block_size])

    return input_transport


def cost_matrix(width, device='cuda'):
    '''transport cost'''
    C = np.zeros([width**2, width**2], dtype=np.float32)

    for m_i in range(width**2):
        i1 = m_i // width
        j1 = m_i % width
        for m_j in range(width**2):
            i2 = m_j // width
            j2 = m_j % width
            C[m_i, m_j] = abs(i1 - i2)**2 + abs(j1 - j2)**2

    C = C / (width - 1)**2
    C = torch.tensor(C)
    if device == 'cuda':
        C = C.cuda()

    return C

cost_matrix_dict = {
    '2': cost_matrix(2, 'cuda').unsqueeze(0),     
    '4': cost_matrix(4, 'cuda').unsqueeze(0),
    '8': cost_matrix(8, 'cuda').unsqueeze(0),
    '16': cost_matrix(16, 'cuda').unsqueeze(0)
    # '2': cost_matrix(2, 'cpu').unsqueeze(0),   
    # '4': cost_matrix(4, 'cpu').unsqueeze(0),
    # '8': cost_matrix(8, 'cpu').unsqueeze(0),
    # '16': cost_matrix(16, 'cpu').unsqueeze(0)    
}

def mask_transport(mask, grad_pool, eps=0.01):
    '''optimal transport plan'''
    batch_size = mask.shape[0]
    block_num = mask.shape[-1]

    n_iter = int(block_num)
    C = cost_matrix_dict[str(block_num)]

    z = (mask > 0).float()
    cost = eps * C - grad_pool.reshape(-1, block_num**2, 1) * z.reshape(-1, 1, block_num**2)

    for _ in range(n_iter):
        row_best = cost.min(-1)[1]
        plan = torch.zeros_like(cost).scatter_(-1, row_best.unsqueeze(-1), 1)

        # column resolve
        cost_fight = plan * cost
        col_best = cost_fight.min(-2)[1]
        plan_win = torch.zeros_like(cost).scatter_(-2, col_best.unsqueeze(-2), 1) * plan
        plan_lose = (1 - plan_win) * plan

        cost += plan_lose

    return plan_win


def graphcut_multi(unary1, unary2, pw_x, pw_y, alpha, beta, eta, n_labels=2, eps=1e-8):
    '''alpha-beta swap algorithm'''
    block_num = unary1.shape[0]

    large_val = 1000 * block_num**2

    if n_labels == 2:
        prior = np.array([-np.log(alpha + eps), -np.log(1 - alpha + eps)])
    elif n_labels == 3:
        prior = np.array([
            -np.log(alpha**2 + eps), -np.log(2 * alpha * (1 - alpha) + eps),
            -np.log((1 - alpha)**2 + eps)
        ])
    elif n_labels == 4:
        prior = np.array([
            -np.log(alpha**3 + eps), -np.log(3 * alpha**2 * (1 - alpha) + eps),
            -np.log(3 * alpha * (1 - alpha)**2 + eps), -np.log((1 - alpha)**3 + eps)
        ])

    prior = eta * prior / block_num**2
    unary_cost = (large_val * np.stack([(1 - lam) * unary1 + lam * unary2 + prior[i]
                                        for i, lam in enumerate(np.linspace(0, 1, n_labels))],
                                       axis=-1)).astype(np.int32)
    pairwise_cost = np.zeros(shape=[n_labels, n_labels], dtype=np.float32)

    for i in range(n_labels):
        for j in range(n_labels):
            pairwise_cost[i, j] = (i - j)**2 / (n_labels - 1)**2

    pw_x = (large_val * (pw_x + beta)).astype(np.int32)
    pw_y = (large_val * (pw_y + beta)).astype(np.int32)
    labels = 1.0 - gco.cut_grid_graph(unary_cost, pairwise_cost, pw_x, pw_y,
                                      algorithm='swap') / (n_labels - 1)
    mask = labels.reshape(block_num, block_num)

    return mask


def neigh_penalty(input1, input2, k):
    '''data local smoothness term'''
    pw_x = input1[:, :, :-1, :] - input2[:, :, 1:, :]
    pw_y = input1[:, :, :, :-1] - input2[:, :, :, 1:]

    pw_x = pw_x[:, :, k - 1::k, :]
    pw_y = pw_y[:, :, :, k - 1::k]

    pw_x = torch.nn.functional.avg_pool2d(pw_x.abs().mean(1), kernel_size=(1, k))
    pw_y = torch.nn.functional.avg_pool2d(pw_y.abs().mean(1), kernel_size=(k, 1))

    return pw_x, pw_y

def mixup_graph(out, y, grad, alpha, lam, index, block_num=2, neigh_size=4, t_size=4, beta=1.2, gamma=0.5, eta=0.2, n_labels=3, t_eps=0.8, transport=False, std=None, mean=None): 
    x1 = out
    x2 = out[index].clone()
    y1=y
    y2=y[index].clone()
    grad1 = grad


    batch_size, _, _, width = x1.shape             
    block_size = width // block_num
    neigh_size = min(neigh_size, block_size)
    t_size = min(t_size, block_size)


    # normalize
    beta = beta / block_num / 16

    # unary term
    grad1_pool = torch.nn.functional.avg_pool2d(grad1, block_size)
    unary1_torch = grad1_pool / grad1_pool.reshape(batch_size, -1).sum(1).reshape(batch_size, 1, 1)
    unary2_torch = unary1_torch[index]

    # calculate pairwise terms
    input1_pool = torch.nn.functional.avg_pool2d(x1 * std + mean, neigh_size)
    input2_pool = input1_pool[index]

    pw_x = torch.zeros([batch_size, 2, 2, block_num - 1, block_num], device='cuda')
    pw_y = torch.zeros([batch_size, 2, 2, block_num, block_num - 1], device='cuda')

    k = block_size // neigh_size

    pw_x[:, 0, 0], pw_y[:, 0, 0] = neigh_penalty(input2_pool, input2_pool, k)
    pw_x[:, 0, 1], pw_y[:, 0, 1] = neigh_penalty(input2_pool, input1_pool, k)
    pw_x[:, 1, 0], pw_y[:, 1, 0] = neigh_penalty(input1_pool, input2_pool, k)
    pw_x[:, 1, 1], pw_y[:, 1, 1] = neigh_penalty(input1_pool, input1_pool, k)

    pw_x = beta * gamma * pw_x
    pw_y = beta * gamma * pw_y

    # re-define unary and pairwise terms to draw graph
    unary1 = unary1_torch.clone()
    unary2 = unary2_torch.clone()

    unary2[:, :-1, :] += (pw_x[:, 1, 0] + pw_x[:, 1, 1]) / 2.
    unary1[:, :-1, :] += (pw_x[:, 0, 1] + pw_x[:, 0, 0]) / 2.
    unary2[:, 1:, :] += (pw_x[:, 0, 1] + pw_x[:, 1, 1]) / 2.
    unary1[:, 1:, :] += (pw_x[:, 1, 0] + pw_x[:, 0, 0]) / 2.

    unary2[:, :, :-1] += (pw_y[:, 1, 0] + pw_y[:, 1, 1]) / 2.
    unary1[:, :, :-1] += (pw_y[:, 0, 1] + pw_y[:, 0, 0]) / 2.
    unary2[:, :, 1:] += (pw_y[:, 0, 1] + pw_y[:, 1, 1]) / 2.
    unary1[:, :, 1:] += (pw_y[:, 1, 0] + pw_y[:, 0, 0]) / 2.

    pw_x = (pw_x[:, 1, 0] + pw_x[:, 0, 1] - pw_x[:, 1, 1] - pw_x[:, 0, 0]) / 2
    pw_y = (pw_y[:, 1, 0] + pw_y[:, 0, 1] - pw_y[:, 1, 1] - pw_y[:, 0, 0]) / 2

    unary1 = unary1.detach().cpu().numpy()
    unary2 = unary2.detach().cpu().numpy()
    pw_x = pw_x.detach().cpu().numpy()
    pw_y = pw_y.detach().cpu().numpy()

    #solve graphcut
    mask = []
    for i in range(batch_size):
        mask.append(graphcut_multi(unary2[i], unary1[i], pw_x[i], pw_y[i], alpha, beta, eta, n_labels))


    # optimal mask
    mask = torch.tensor(mask, dtype=torch.float32, device='cuda')
    mask = mask.unsqueeze(1)


    # tranport
    if transport:
        if t_size == -1:
            t_block_num = block_num
            t_size = block_size
        elif t_size < block_size:
            # block_size % t_size should be 0
            t_block_num = width // t_size
            mask = torch.nn.functional.interpolate(mask, size=t_block_num)
            grad1_pool = torch.nn.functional.avg_pool2d(grad1, t_size)
            unary1_torch = grad1_pool / grad1_pool.reshape(batch_size, -1).sum(1).reshape(
                batch_size, 1, 1)
            unary2_torch = unary1_torch[index]
        else:
            t_block_num = block_num

        plan = mask_transport(mask, unary1_torch, eps=t_eps)
        x1 = transport_image(x1, plan, batch_size, t_block_num, t_size)

        plan = mask_transport(1 - mask, unary2_torch, eps=t_eps)
        x2 = transport_image(x2, plan, batch_size, t_block_num, t_size)

    # final mask and mixed ratio
    mask = torch.nn.functional.interpolate(mask, size=width)
    ratio = mask.reshape(batch_size, -1).mean(-1)


    """
    mask.shape: torch.Size([4, 1, 32, 32])
    ratio: tensor([0.5000, 0.6250, 0.3750, 0.5000], device='cuda:0')
    ratio.shape: torch.Size([4])
    """


    """
    x1.shape: torch.Size([4, 3, 32, 32])
    x2.shape: torch.Size([4, 3, 32, 32])
    """


    mixed_x = mask * x1 + (1 - mask) * x2
    mixed_y = ratio.unsqueeze(-1) * y1 + (1 - ratio.unsqueeze(-1)) * y2

    return mixed_x, mixed_y