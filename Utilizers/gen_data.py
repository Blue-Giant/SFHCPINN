import torch
import numpy as np
import scipy.stats.qmc as stqmc


def rand_bd_inside(batch_size, variable_dim, region_a, region_b, init_l, init_r, to_torch=True, to_float=True,
                   to_cuda=False, gpu_no=0, use_grad=True, opt2sampler='lhs'):
    # 生成内部点
    assert (int(variable_dim) == 2)
    if opt2sampler == 'random':
        x_inside = (region_b - region_a) * np.random.random([batch_size, 1]) + region_a
        t_inside = (init_r - init_l) * np.random.random([batch_size, 1]) + init_l
        inside_points = np.concatenate([x_inside, t_inside], axis=-1)
    else:
        sampler = stqmc.LatinHypercube(d=1)
        # temp1 = sampler.random(batch_size)
        # temp2 = sampler.random(batch_size)
        x_inside = (region_b - region_a) * sampler.random(batch_size) + region_a
        t_inside = (init_r - init_l) * sampler.random(batch_size) + init_l
        inside_points = np.concatenate([x_inside, t_inside], axis=-1)

    # inside_points = np.random.random([batch_size, 2])
    # inside_points[:, 0] = region_a + inside_points[:, 0] * (region_b - region_a)
    # inside_points[:, 1] = init_l + inside_points[:, 1] * (init_r - init_l)

    if to_float:
        inside_points = inside_points.astype(np.float32)

    if to_torch:
        inside_points = torch.from_numpy(inside_points)
        if to_cuda:
            inside_points = inside_points.cuda(device='cuda:' + str(gpu_no))
        inside_points.requires_grad = use_grad

    return inside_points


def rand_bd(batch_size, variable_dim, region_a, region_b, init_l, init_r, to_torch=True, to_float=True, to_cuda=False,
            gpu_no=0, use_grad=True):
    # 全边界在Hard_PINN中没什么用
    region_a = float(region_a)
    region_b = float(region_b)
    init_l = float(init_l)
    init_r = float(init_r)
    assert (int(variable_dim) == 2)
    bd1 = int(batch_size * np.random.random(1))
    res = batch_size - bd1
    bd2 = int(res * np.random.random(1))
    res = res - bd2
    bd3 = int(res * np.random.random(1))
    bd4 = res - bd3
    x_left_bd = (init_r - init_l) * np.random.random([bd1, 2]) + init_l  # 浮点数都是从0-1中随机。
    x_right_bd = (init_r - init_l) * np.random.random([bd2, 2]) + init_l
    y_bottom_bd = (region_b - region_a) * np.random.random([bd3, 2]) + region_a
    y_top_bd = (region_b - region_a) * np.random.random([bd3, 2]) + region_a

    x_left_bd[:, 0] = region_a
    x_right_bd[:, 0] = region_b
    y_bottom_bd[:, 1] = init_l
    y_top_bd[:, 1] = init_r

    if to_float:
        x_left_bd = x_left_bd.astype(np.float32)
        x_right_bd = x_right_bd.astype(np.float32)
        y_bottom_bd = y_bottom_bd.astype(np.float32)
        y_top_bd = y_top_bd.astype(np.float32)
    if to_torch:
        x_left_bd = torch.from_numpy(x_left_bd)
        x_right_bd = torch.from_numpy(x_right_bd)
        y_bottom_bd = torch.from_numpy(y_bottom_bd)
        y_top_bd = torch.from_numpy(y_top_bd)
        if to_cuda:
            x_left_bd = x_left_bd.cuda(device='cuda:' + str(gpu_no))
            x_right_bd = x_right_bd.cuda(device='cuda:' + str(gpu_no))
            y_bottom_bd = y_bottom_bd.cuda(device='cuda:' + str(gpu_no))
            y_top_bd = y_top_bd.cuda(device='cuda:' + str(gpu_no))

        x_left_bd.requires_grad = use_grad
        x_right_bd.requires_grad = use_grad
        y_bottom_bd.requires_grad = use_grad
        y_top_bd.requires_grad = use_grad

    return x_left_bd, x_right_bd, y_bottom_bd, y_top_bd


def rand_bd_inflow(batch_size, variable_dim, region_a, region_b, init_l, init_r, to_torch=True, to_float=True,
                   to_cuda=False, gpu_no=0, use_grad=False, opt2sampler='lhs'):
    # 有Dirichlet的边界 叫做INFLOW边界
    assert (int(variable_dim) == 2)
    # bd1 = int(batch_size * np.random.random(1))
    # bd2 = batch_size - bd1
    if opt2sampler == 'random':
        x_right_bd = (init_r - init_l) * np.random.random([batch_size, 2]) + init_l
        t_start_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
    else:
        sampler = stqmc.LatinHypercube(d=1)
        x_right_bd = (init_r - init_l) * sampler.random(batch_size) + init_l
        t_start_bd = (region_b - region_a) * sampler.random(batch_size) + region_a

    x_right_bd[:, 0] = region_b
    t_start_bd[:, 1] = init_l
    inflow_boundary_points = np.concatenate([x_right_bd, t_start_bd], 0)
    if to_float:
        inflow_boundary_points = inflow_boundary_points.astype(np.float32)

    if to_torch:
        inflow_boundary_points = torch.from_numpy(inflow_boundary_points)
        if to_cuda:
            inflow_boundary_points = inflow_boundary_points(device='cuda:' + str(gpu_no))
        inflow_boundary_points.requires_grad = use_grad

    return inflow_boundary_points


# 两边均为Dirichlet边界时的边界点生成。对于时空Advection diffusion equation 而言，1维情形下：左右均为迪利克雷边界，
# 那么意味着初始条件和边界条件可以用
def rand_bd_inflow2double_dirichlet(batch_size, variable_dim, region_a, region_b, init_begin, init_end, to_torch=True,
                                    to_float=True, to_cuda=False, gpu_no=0, use_grad=False, opt2sampler='lhs'):
    # 有Dirichlet的边界 叫做INFLOW边界
    # region_a = float(region_a)
    # region_b = float(region_b)
    # init_l = float(init_l)
    # init_r = float(init_r)
    assert (int(variable_dim) == 2)

    if str.lower(opt2sampler) == 'random':
        t2left_right = (init_end - init_begin) * np.random.random([batch_size, 1]) + init_begin

        x2left = np.ones([batch_size, 1]) * region_a
        xt_left = np.concatenate([x2left, t2left_right], axis=-1)

        x2right = np.ones([batch_size, 1]) * region_b
        xt_right = np.concatenate([x2right, t2left_right], axis=-1)

        x2begin_end = (region_b - region_a) * np.random.random([batch_size, 1]) + region_a
        t2begin = np.ones([batch_size, 1]) * init_begin
        xt_begin = np.concatenate([x2begin_end, t2begin], axis=-1)
    else:
        sampler2time = stqmc.LatinHypercube(d=1)
        sampler2x = stqmc.LatinHypercube(d=1)
        t2left_right = (init_end - init_begin) * sampler2time.random(batch_size) + init_begin

        x2left = np.ones([batch_size, 1]) * region_a
        xt_left = np.concatenate([x2left, t2left_right], axis=-1)

        x2right = np.ones([batch_size, 1]) * region_b
        xt_right = np.concatenate([x2right, t2left_right], axis=-1)

        x2begin_end = (region_b - region_a) * sampler2x.random(batch_size) + region_a
        t2begin = np.ones([batch_size, 1]) * init_begin
        xt_begin = np.concatenate([x2begin_end, t2begin], axis=-1)

    # t2end = np.ones([batch_size, 1]) * init_l
    # xt_end = np.concatenate([x2begin_end, t2end], axis=-1)

    inflow_boundary_points = np.concatenate([xt_left, xt_right, xt_begin], 0)
    if to_float:
        inflow_boundary_points = inflow_boundary_points.astype(np.float32)

    if to_torch:
        inflow_boundary_points = torch.from_numpy(inflow_boundary_points)
        if to_cuda:
            inflow_boundary_points = inflow_boundary_points(device='cuda:' + str(gpu_no))
        inflow_boundary_points.requires_grad = use_grad

    return inflow_boundary_points


# 两边均为Neumann 边界时的边界点生成。对于时空Advection diffusion equation 而言，1维情形下：左右均为诺伊曼边界，那么意味着初始条件可以用
def rand_inflow2double_neumann_bd(batch_size, variable_dim, region_a, region_b, init_l, init_r, to_torch=True,
                                  to_float=True, to_cuda=False, gpu_no=0, use_grad=True, opt2sampler='lhs'):
    '''
    这个函数主要用于生成诺伊曼问题的边界点
    Args:
        batch_size:
        region_a:
        region_b:
        init_l:
        init_r:
        to_torch:
        to_float:
        to_cuda:
        gpu_no:
        use_grad:
    Returns:

    '''
    if str.lower(opt2sampler) == 'random':
        inflow_points = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
    else:
        sampler = stqmc.LatinHypercube(d=2)
        inflow_points = (region_b - region_a) * sampler.random(batch_size) + region_a
    inflow_points[:, 1] = init_l

    if to_float:
        inflow_points = inflow_points.astype(np.float32)

    if to_torch:
        inflow_points = torch.from_numpy(inflow_points)
        if to_cuda:
            inflow_points = inflow_points.cuda(device='cuda:' + str(gpu_no))
        inflow_points.requires_grad = use_grad
    return inflow_points


def rand_nm_bd(batch_size, variable_dim, region_a, init_l, init_r, to_torch=True, to_float=True, to_cuda=False,
               gpu_no=0,
               use_grad=True):
    # 生成诺依曼边界
    region_a = float(region_a)
    init_l = float(init_l)
    init_r = float(init_r)
    assert (int(variable_dim) == 2)
    x_left_bd = (init_r - init_l) * np.random.random([batch_size, 2]) + init_l
    x_left_bd[:, 0] = region_a
    if to_float:
        x_left_bd = x_left_bd.astype(np.float32)

    if to_torch:
        x_left_bd = torch.from_numpy(x_left_bd)
        if to_cuda:
            x_left_bd = x_left_bd(device='cuda:' + str(gpu_no))
        x_left_bd.requires_grad = use_grad

    return x_left_bd


def rand_bd_inflow_EX6(batch_size, region_a, region_b, init_l, init_r, to_torch=True, to_float=True, to_cuda=False,
                       gpu_no=0,
                       use_grad=True):
    # 生成Dirhlet 边界的点
    region_a = float(region_a)
    region_b = float(region_b)
    init_l = float(init_l)
    init_r = float(init_r)
    y_bottom_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
    y_bottom_bd[:, 1] = init_l
    inflow_boundary_points = y_bottom_bd
    if to_float:
        inflow_boundary_points = inflow_boundary_points.astype(np.float32)

    if to_torch:
        inflow_boundary_points = torch.from_numpy(inflow_boundary_points)
        if to_cuda:
            inflow_boundary_points = inflow_boundary_points(device='cuda:' + str(gpu_no))
        inflow_boundary_points.requires_grad = use_grad

    return inflow_boundary_points


def rand_bd_inside_EX6(batch_size, region_a, region_b, init_l, init_r, to_torch=True, to_float=True,
                   to_cuda=False, gpu_no=0, use_grad=True):
    # 生成二维矩阵的内部点
    region_a = float(region_a)
    region_b = float(region_b)
    init_l = float(init_l)
    init_r = float(init_r)
    inside_points = np.random.random([batch_size, 2])
    inside_points[:, 0] = region_a + inside_points[:, 0] * (region_b - region_a)
    inside_points[:, 1] = init_l + inside_points[:, 1] * (init_r - init_l)

    if to_float:
        inside_points = inside_points.astype(np.float32)

    if to_torch:
        inside_points = torch.from_numpy(inside_points)
        if to_cuda:
            inside_points = inside_points.cuda(device='cuda:' + str(gpu_no))
        inside_points.requires_grad = use_grad

    return inside_points