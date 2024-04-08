import numpy as np
import torch
import scipy.stats.qmc as stqmc


def load_data2porous_domain(region_left=0.0, region_right=0.0, region_bottom=0.0, region_top=0.0,
                            to_torch=True, to_float=True, to_cuda=False, gpu_no=0, use_grad2x=False, scale_trans=False,
                            scale2x=2.0, scale2y=2.0, base2x=0.0, base2y=0.0):
    data_path = '../data2PorousDomain_2D/Normalized/xy_porous5.txt'
    porous_points2xy = np.loadtxt(data_path)
    porous_points2xy = porous_points2xy.astype(dtype=np.float32)
    shape2data = np.shape(porous_points2xy)
    num2points = shape2data[0]
    points = []
    for ip in range(num2points):
        point = np.reshape(porous_points2xy[ip], newshape=(-1, 2))
        if point[0, 0] == region_left or point[0, 0] == region_right:
            continue
        elif point[0, 1] == region_bottom or point[0, 1] == region_top:
            continue
        else:
            points.append(point)
    xy_inside = np.concatenate(points, axis=0)
    if scale_trans:
        xy_inside[:, 0:1] = scale2x * xy_inside[:, 0:1] + base2x
        xy_inside[:, 1:2] = scale2y * xy_inside[:, 1:2] + base2y

    if to_torch:
        xy_inside = torch.from_numpy(xy_inside)

        if to_cuda:
            xy_inside = xy_inside.cuda(device='cuda:' + str(gpu_no))

        xy_inside.requires_grad = use_grad2x

    return xy_inside


# ---------------------------------------------- 数据集的生成 ---------------------------------------------------
#  内部生成，方形区域[a,b]^n生成随机数, n代表变量个数，使用随机采样方法
def rand_it(batch_size, variable_dim, region_a, region_b, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
            use_grad2x=False):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    x_it = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    if to_float:
        x_it = x_it.astype(np.float32)

    if to_torch:
        x_it = torch.from_numpy(x_it)

        if to_cuda:
            x_it = x_it.cuda(device='cuda:' + str(gpu_no))

        x_it.requires_grad = use_grad2x

    return x_it


#  内部生成,矩形区域
def rand_in_1D(batch_size=100, variable_dim=1, region_a=0.0, region_b=1.0, to_torch=True, to_float=True,
               to_cuda=False, gpu_no=0, use_grad2x=False, opt2sampler='lhs'):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    assert 1 == int(variable_dim)
    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=variable_dim)
        x_it = (region_b - region_a) * sampler.random(batch_size) + region_a
    else:
        x_it = (region_b - region_a) * np.random.rand(batch_size, 1) + region_a

    if to_float:
        x_it = x_it.astype(np.float32)

    if to_torch:
        x_it = torch.from_numpy(x_it)

        if to_cuda:
            x_it = x_it.cuda(device='cuda:' + str(gpu_no))

        x_it.requires_grad = use_grad2x

    return x_it


#  内部生成, 矩形区域, 使用随机采样方法
def rand_in_2D(batch_size=100, variable_dim=2, region_left=0.0, region_right=1.0, region_bottom=0.0, region_top=1.0,
               to_torch=True, to_float=True, to_cuda=False, gpu_no=0, use_grad2x=False, opt2sampler='lhs'):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    assert 2 == int(variable_dim)
    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x_it = (region_right - region_left) * sampler.random(batch_size) + region_left
        y_it = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom
    else:
        x_it = (region_right - region_left) * np.random.rand(batch_size, 1) + region_left
        y_it = (region_top - region_bottom) * np.random.rand(batch_size, 1) + region_bottom

    xy_in = np.concatenate([x_it, y_it], axis=-1)
    if to_float:
        xy_in = xy_in.astype(np.float32)

    if to_torch:
        xy_in = torch.from_numpy(xy_in)

        if to_cuda:
            xy_in = xy_in.cuda(device='cuda:' + str(gpu_no))

        xy_in.requires_grad = use_grad2x

    return xy_in


#  内部生成, 矩形区域, 使用随机采样方法
def rand_in_3D(batch_size=100, variable_dim=3, region_left=0.0, region_right=1.0, region_behind=0.0, region_front=1.0,
               region_bottom=0.0, region_top=1.0, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
               use_grad2x=False, opt2sampler='lhs'):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    assert 3 == int(variable_dim)
    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x_it = (region_right - region_left) * sampler.random(batch_size) + region_left
        y_it = (region_front - region_behind) * sampler.random(batch_size) + region_behind
        z_it = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom
    else:
        x_it = (region_right - region_left) * np.random.rand(batch_size, 1) + region_left
        y_it = (region_front - region_behind) * np.random.rand(batch_size, 1) + region_behind
        z_it = (region_top - region_bottom) * np.random.rand(batch_size, 1) + region_bottom

    xyz_in = np.concatenate([x_it, y_it, z_it], axis=-1)
    if to_float:
        xyz_in = xyz_in.astype(np.float32)

    if to_torch:
        xyz_in = torch.from_numpy(xyz_in)

        if to_cuda:
            xyz_in = xyz_in.cuda(device='cuda:' + str(gpu_no))

        xyz_in.requires_grad = use_grad2x

    return xyz_in


#  内部生成,矩形区域, 使用随机采样方法
def rand_in_4D(batch_size=100, variable_dim=3, region_xleft=0.0, region_xright=1.0, region_yleft=0.0, region_yright=1.0,
               region_zleft=0.0, region_zright=1.0, region_sleft=0.0, region_sright=1.0, to_torch=True, to_float=True,
               to_cuda=False, gpu_no=0, use_grad2x=False, opt2sampler='lhs'):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    assert 4 == int(variable_dim)
    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x_it = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y_it = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z_it = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s_it = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
    else:
        x_it = (region_xright - region_xleft) * np.random.rand(batch_size, 1) + region_xleft
        y_it = (region_yright - region_yleft) * np.random.rand(batch_size, 1) + region_yleft
        z_it = (region_zright - region_zleft) * np.random.rand(batch_size, 1) + region_zleft
        s_it = (region_sright - region_sleft) * np.random.rand(batch_size, 1) + region_sleft

    xyzs_in = np.concatenate([x_it, y_it, z_it, s_it], axis=-1)
    if to_float:
        xyzs_in = xyzs_in.astype(np.float32)

    if to_torch:
        xyzs_in = torch.from_numpy(xyzs_in)

        if to_cuda:
            xyzs_in = xyzs_in.cuda(device='cuda:' + str(gpu_no))

        xyzs_in.requires_grad = use_grad2x

    return xyzs_in


#  内部生成,矩形区域, 使用随机采样方法
def rand_in_5D(batch_size=100, variable_dim=3, region_xleft=0.0, region_xright=1.0, region_yleft=0.0, region_yright=1.0,
               region_zleft=0.0, region_zright=1.0, region_sleft=0.0, region_sright=1.0, region_tleft=0.0,
               region_tright=1.0, to_torch=True, to_float=True, to_cuda=False, gpu_no=0, use_grad2x=False,
               opt2sampler='lhs'):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    assert 5 == int(variable_dim)
    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x_it = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y_it = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z_it = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s_it = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        t_it = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
    else:
        x_it = (region_xright - region_xleft) * np.random.rand(batch_size, 1) + region_xleft
        y_it = (region_yright - region_yleft) * np.random.rand(batch_size, 1) + region_yleft
        z_it = (region_zright - region_zleft) * np.random.rand(batch_size, 1) + region_zleft
        s_it = (region_sright - region_sleft) * np.random.rand(batch_size, 1) + region_sleft
        t_it = (region_tright - region_tleft) * np.random.rand(batch_size, 1) + region_tleft

    xyzst_in = np.concatenate([x_it, y_it, z_it, s_it, t_it], axis=-1)
    if to_float:
        xyzst_in = xyzst_in.astype(np.float32)

    if to_torch:
        xyzst_in = torch.from_numpy(xyzst_in)

        if to_cuda:
            xyzst_in = xyzst_in.cuda(device='cuda:' + str(gpu_no))

        xyzst_in.requires_grad = use_grad2x

    return xyzst_in


#  内部生成,矩形区域, 使用随机采样方法
def rand_in_8D(batch_size=100, variable_dim=3, region_xleft=0.0, region_xright=1.0, region_yleft=0.0, region_yright=1.0,
               region_zleft=0.0, region_zright=1.0, region_sleft=0.0, region_sright=1.0, region_tleft=0.0,
               region_tright=1.0, region_pleft=0.0, region_pright=1.0, region_qleft=0.0, region_qright=1.0,
               region_rleft=0.0, region_rright=1.0, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
               use_grad2x=False, opt2sampler='lhs'):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    assert 8 == int(variable_dim)
    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x_it = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y_it = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z_it = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s_it = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        t_it = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        p_it = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        q_it = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        r_it = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft
    else:
        x_it = (region_xright - region_xleft) * np.random.rand(batch_size, 1) + region_xleft
        y_it = (region_yright - region_yleft) * np.random.rand(batch_size, 1) + region_yleft
        z_it = (region_zright - region_zleft) * np.random.rand(batch_size, 1) + region_zleft
        s_it = (region_sright - region_sleft) * np.random.rand(batch_size, 1) + region_sleft
        t_it = (region_tright - region_tleft) * np.random.rand(batch_size, 1) + region_tleft
        p_it = (region_pright - region_pleft) * np.random.rand(batch_size, 1) + region_pleft
        q_it = (region_qright - region_qleft) * np.random.rand(batch_size, 1) + region_qleft
        r_it = (region_rright - region_rleft) * np.random.rand(batch_size, 1) + region_rleft

    xyzstpqr_in = np.concatenate([x_it, y_it, z_it, s_it, t_it, p_it, q_it, r_it], axis=-1)
    if to_float:
        xyzstpqr_in = xyzstpqr_in.astype(np.float32)

    if to_torch:
        xyzstpqr_in = torch.from_numpy(xyzstpqr_in)

        if to_cuda:
            xyzstpqr_in = xyzstpqr_in.cuda(device='cuda:' + str(gpu_no))

        xyzstpqr_in.requires_grad = use_grad2x

    return xyzstpqr_in


#  内部生成, 方形区域[a,b]^n生成随机数, n代表变量个数. 使用Sobol采样方法
def rand_it_Sobol(batch_size, variable_dim, region_a, region_b, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
                   use_grad2x=False):
    sampler = stqmc.Sobol(d=variable_dim, scramble=True)
    x_it = (region_b - region_a) * sampler.random(n=batch_size) + region_a
    if to_float:
        x_it = x_it.astype(np.float32)

    if to_torch:
        x_it = torch.from_numpy(x_it)

        if to_cuda:
            x_it = x_it.cuda(device='cuda:' + str(gpu_no))

        x_it.requires_grad = use_grad2x

    return x_it


# 边界生成点
def rand_bd_1D(batch_size=100, variable_dim=1, region_a=0.0, region_b=1.0, to_torch=True, to_float=True,
               to_cuda=False, gpu_no=0, use_grad2x=False):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    assert (int(variable_dim) == 1)

    region_a = float(region_a)
    region_b = float(region_b)

    x_left_bd = np.ones(shape=[batch_size, 1], dtype=np.float32) * region_a
    x_right_bd = np.ones(shape=[batch_size, 1], dtype=np.float32) * region_b
    if to_float:
        x_left_bd = x_left_bd.astype(np.float32)
        x_right_bd = x_right_bd.astype(np.float32)

    if to_torch:
        x_left_bd = torch.from_numpy(x_left_bd)
        x_right_bd = torch.from_numpy(x_right_bd)

        if to_cuda:
            x_left_bd = x_left_bd.cuda(device='cuda:' + str(gpu_no))
            x_right_bd = x_right_bd.cuda(device='cuda:' + str(gpu_no))

        x_left_bd.requires_grad = use_grad2x
        x_right_bd.requires_grad = use_grad2x

    return x_left_bd, x_right_bd


#  内部生成，方形区域[a,b]^n生成随机数, n代表变量个数，使用随机采样方法
def rand_bd_2D(batch_size=1000, variable_dim=2, region_left=0.0, region_right=1.0, region_bottom=0.0, region_top=1.0,
               to_torch=False, to_float=True, to_cuda=False, gpu_no=0, use_grad=False, opt2sampler='lhs'):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    # np.random.random((100, 50)) 上方代表生成100行 50列的随机浮点数，浮点数范围 : (0,1)
    # np.random.random([100, 50]) 和 np.random.random((100, 50)) 效果一样

    assert (int(variable_dim) == 2)

    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=variable_dim)
        x_left_bd = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom
        x_right_bd = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom
        y_bottom_bd = (region_right - region_left) * sampler.random(batch_size) + region_left
        y_top_bd = (region_right - region_left) * sampler.random(batch_size) + region_left
    else:
        x_left_bd = (region_top - region_bottom) * np.random.random([batch_size, 2]) + region_bottom  # 浮点数都是从0-1中随机。
        x_right_bd = (region_top - region_bottom) * np.random.random([batch_size, 2]) + region_bottom
        y_bottom_bd = (region_right - region_left) * np.random.random([batch_size, 2]) + region_left
        y_top_bd = (region_right - region_left) * np.random.random([batch_size, 2]) + region_left

    for ii in range(batch_size):
        x_left_bd[ii, 0] = region_left
        x_right_bd[ii, 0] = region_right
        y_bottom_bd[ii, 1] = region_bottom
        y_top_bd[ii, 1] = region_top

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


#  内部生成, 方形区域[a,b]^n生成随机数, n代表变量个数. 使用Sobol采样方法
def rand_bd_2D_sobol(batch_size, variable_dim, region_a, region_b, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
                     use_grad=False):
    region_a = float(region_a)
    region_b = float(region_b)
    assert (int(variable_dim) == 2)
    sampler = stqmc.Sobol(d=variable_dim, scramble=True)
    x_left_bd = (region_b-region_a) * sampler.random(n=batch_size) + region_a   # 浮点数都是从0-1中随机。
    x_right_bd = (region_b - region_a) * sampler.random(n=batch_size) + region_a
    y_bottom_bd = (region_b - region_a) * sampler.random(n=batch_size) + region_a
    y_top_bd = (region_b - region_a) * sampler.random(n=batch_size) + region_a
    for ii in range(batch_size):
        x_left_bd[ii, 0] = region_a
        x_right_bd[ii, 0] = region_b
        y_bottom_bd[ii, 1] = region_a
        y_top_bd[ii, 1] = region_b

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


def rand_bd_3D(batch_size=1000, variable_dim=3, region_left=0.0, region_right=1.0, region_behind=0.0, region_front=1.0,
               region_bottom=0.0, region_top=1.0, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
               use_grad=False, opt2sampler='lhs'):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    assert (int(variable_dim) == 3)
    x_left_bd = np.zeros(shape=[batch_size, variable_dim])
    x_right_bd = np.zeros(shape=[batch_size, variable_dim])
    y_behind_bd = np.zeros(shape=[batch_size, variable_dim])
    y_front_bd = np.zeros(shape=[batch_size, variable_dim])
    z_bottom_bd = np.zeros(shape=[batch_size, variable_dim])
    z_top_bd = np.zeros(shape=[batch_size, variable_dim])

    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x_left_bd[:, 0:1] = region_left
        x_left_bd[:, 1:2] = (region_front - region_behind) * sampler.random(batch_size) + region_behind
        x_left_bd[:, 2:3] = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom

        x_right_bd[:, 0:1] = region_right
        x_right_bd[:, 1:2] = (region_front - region_behind) * sampler.random(batch_size) + region_behind
        x_right_bd[:, 2:3] = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom

        y_behind_bd[:, 0:1] = (region_right - region_left) * sampler.random(batch_size) + region_left
        y_behind_bd[:, 1:2] = region_behind
        y_behind_bd[:, 2:3] = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom

        y_front_bd[:, 0:1] = (region_right - region_left) * sampler.random(batch_size) + region_left
        y_front_bd[:, 1:2] = region_front
        y_front_bd[:, 2:3] = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom

        z_bottom_bd[:, 0:1] = (region_right - region_left) * sampler.random(batch_size) + region_left
        z_bottom_bd[:, 1:2] = (region_front - region_behind) * sampler.random(batch_size) + region_behind
        z_bottom_bd[:, 2:3] = region_bottom

        z_top_bd[:, 0:1] = (region_right - region_left) * sampler.random(batch_size) + region_left
        z_top_bd[:, 1:2] = (region_front - region_behind) * sampler.random(batch_size) + region_behind
        z_top_bd[:, 2:3] = region_top
    else:
        x_left_bd[:, 0:1] = region_left
        x_left_bd[:, 1:2] = (region_front - region_behind) * np.random.random([batch_size, 1]) + region_behind
        x_left_bd[:, 2:3] = (region_top - region_bottom) * np.random.random([batch_size, 1]) + region_bottom

        x_right_bd[:, 0:1] = region_right
        x_right_bd[:, 1:2] = (region_front - region_behind) * np.random.random([batch_size, 1]) + region_behind
        x_right_bd[:, 2:3] = (region_top - region_bottom) * np.random.random([batch_size, 1]) + region_bottom

        y_behind_bd[:, 0:1] = (region_right - region_left) * np.random.random([batch_size, 1]) + region_left
        y_behind_bd[:, 1:2] = region_behind
        y_behind_bd[:, 2:3] = (region_top - region_bottom) * np.random.random([batch_size, 1]) + region_bottom

        y_front_bd[:, 0:1] = (region_right - region_left) * np.random.random([batch_size, 1]) + region_left
        y_front_bd[:, 1:2] = region_front
        y_front_bd[:, 2:3] = (region_top - region_bottom) * np.random.random([batch_size, 1]) + region_bottom

        z_bottom_bd[:, 0:1] = (region_right - region_left) * np.random.random([batch_size, 1]) + region_left
        z_bottom_bd[:, 1:2] = (region_front - region_behind) * np.random.random([batch_size, 1]) + region_behind
        z_bottom_bd[:, 2:3] = region_bottom

        z_top_bd[:, 0:1] = (region_right - region_left) * np.random.random([batch_size, 1]) + region_left
        z_top_bd[:, 1:2] = (region_front - region_behind) * np.random.random([batch_size, 1]) + region_behind
        z_top_bd[:, 2:3] = region_top

    if to_float:
        z_bottom_bd = z_bottom_bd.astype(np.float32)
        z_top_bd = z_top_bd.astype(np.float32)
        x_left_bd = x_left_bd.astype(np.float32)
        x_right_bd = x_right_bd.astype(np.float32)
        y_front_bd = y_front_bd.astype(np.float32)
        y_behind_bd = y_behind_bd.astype(np.float32)
    if to_torch:
        z_bottom_bd = torch.from_numpy(z_bottom_bd)
        z_top_bd = torch.from_numpy(z_top_bd)
        x_left_bd = torch.from_numpy(x_left_bd)
        x_right_bd = torch.from_numpy(x_right_bd)
        y_front_bd = torch.from_numpy(y_front_bd)
        y_behind_bd = torch.from_numpy(y_behind_bd)

        if to_cuda:
            z_bottom_bd = z_bottom_bd.cuda(device='cuda:' + str(gpu_no))
            z_top_bd = z_top_bd.cuda(device='cuda:' + str(gpu_no))
            x_left_bd = x_left_bd.cuda(device='cuda:' + str(gpu_no))
            x_right_bd = x_right_bd.cuda(device='cuda:' + str(gpu_no))
            y_front_bd = y_front_bd.cuda(device='cuda:' + str(gpu_no))
            y_behind_bd = y_behind_bd.cuda(device='cuda:' + str(gpu_no))

        z_bottom_bd.requires_grad = use_grad
        z_top_bd.requires_grad = use_grad
        x_left_bd.requires_grad = use_grad
        x_right_bd.requires_grad = use_grad
        y_front_bd.requires_grad = use_grad
        y_behind_bd.requires_grad = use_grad

    return x_left_bd, x_right_bd, y_front_bd, y_behind_bd, z_bottom_bd, z_top_bd


def rand_bd_4D(batch_size=1000, variable_dim=3, region_xleft=0.0, region_xright=1.0, region_yleft=0.0,
               region_yright=1.0, region_zleft=0.0, region_zright=1.0, region_sleft=0.0, region_sright=1.0,
               to_torch=True, to_float=True, to_cuda=False, gpu_no=0, use_grad=False, opt2sampler='lhs'):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    assert (int(variable_dim) == 4)
    x00_bd = np.zeros(shape=[batch_size, variable_dim])
    x01_bd = np.zeros(shape=[batch_size, variable_dim])
    y00_bd = np.zeros(shape=[batch_size, variable_dim])
    y01_bd = np.zeros(shape=[batch_size, variable_dim])
    z00_bd = np.zeros(shape=[batch_size, variable_dim])
    z01_bd = np.zeros(shape=[batch_size, variable_dim])
    s00_bd = np.zeros(shape=[batch_size, variable_dim])
    s01_bd = np.zeros(shape=[batch_size, variable_dim])

    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x00_bd[:, 0:1] = region_xleft
        x00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        x00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        x00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft

        x01_bd[:, 0:1] = region_xright
        x01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        x01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        x01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft

        y00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y00_bd[:, 1:2] = region_yleft
        y00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        y00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft

        y01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y01_bd[:, 1:2] = region_yright
        y01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        y01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft

        z00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        z00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z00_bd[:, 2:3] = region_zleft
        z00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft

        z01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        z01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z01_bd[:, 2:3] = region_zright
        z01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft

        s00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        s00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        s00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s00_bd[:, 3:4] = region_sleft

        s01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        s01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        s01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s01_bd[:, 3:4] = region_sright
    else:
        sampler = stqmc.LatinHypercube(d=1)
        x00_bd[:, 0:1] = region_xleft
        x00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        x00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        x00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft

        x01_bd[:, 0:1] = region_xright
        x01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        x01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        x01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft

        y00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        y00_bd[:, 1:2] = region_yleft
        y00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        y00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft

        y01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        y01_bd[:, 1:2] = region_yright
        y01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        y01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft

        z00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        z00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        z00_bd[:, 2:3] = region_zleft
        z00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft

        z01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        z01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        z01_bd[:, 2:3] = region_zright
        z01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft

        s00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        s00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        s00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        s00_bd[:, 3:4] = region_sleft

        s01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        s01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        s01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        s01_bd[:, 3:4] = region_sright

    if to_float:
        x00_bd = x00_bd.astype(np.float32)
        x01_bd = x01_bd.astype(np.float32)

        y00_bd = y00_bd.astype(np.float32)
        y01_bd = y01_bd.astype(np.float32)

        z00_bd = z00_bd.astype(np.float32)
        z01_bd = z01_bd.astype(np.float32)

        s00_bd = s00_bd.astype(np.float32)
        s01_bd = s01_bd.astype(np.float32)

    if to_torch:
        x00_bd = torch.from_numpy(x00_bd)
        x01_bd = torch.from_numpy(x01_bd)

        y00_bd = torch.from_numpy(y00_bd)
        y01_bd = torch.from_numpy(y01_bd)

        z00_bd = torch.from_numpy(z00_bd)
        z01_bd = torch.from_numpy(z01_bd)

        s00_bd = torch.from_numpy(s00_bd)
        s01_bd = torch.from_numpy(s01_bd)

        if to_cuda:
            x00_bd = x00_bd.cuda(device='cuda:' + str(gpu_no))
            x01_bd = x01_bd.cuda(device='cuda:' + str(gpu_no))

            y00_bd = y00_bd.cuda(device='cuda:' + str(gpu_no))
            y01_bd = y01_bd.cuda(device='cuda:' + str(gpu_no))

            z00_bd = z00_bd.cuda(device='cuda:' + str(gpu_no))
            z01_bd = z01_bd.cuda(device='cuda:' + str(gpu_no))

            s00_bd = s00_bd.cuda(device='cuda:' + str(gpu_no))
            s01_bd = s01_bd.cuda(device='cuda:' + str(gpu_no))

        x00_bd.requires_grad = use_grad
        x01_bd.requires_grad = use_grad
        y00_bd.requires_grad = use_grad
        y01_bd.requires_grad = use_grad
        z00_bd.requires_grad = use_grad
        z01_bd.requires_grad = use_grad
        s00_bd.requires_grad = use_grad
        s01_bd.requires_grad = use_grad

    return x00_bd, x01_bd, y00_bd, y01_bd, z00_bd, z01_bd, s00_bd, s01_bd


def rand_bd_5D(batch_size=1000, variable_dim=5, region_xleft=0.0, region_xright=1.0, region_yleft=0.0, region_yright=1.0,
               region_zleft=0.0, region_zright=1.0, region_sleft=0.0, region_sright=1.0, region_tleft=0.0,
               region_tright=1.0, to_torch=True, to_float=True, to_cuda=False, gpu_no=0, use_grad=False,
               opt2sampler='lhs'):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    assert variable_dim == 5
    x00_bd = np.zeros(shape=[batch_size, variable_dim])
    x01_bd = np.zeros(shape=[batch_size, variable_dim])

    y00_bd = np.zeros(shape=[batch_size, variable_dim])
    y01_bd = np.zeros(shape=[batch_size, variable_dim])

    z00_bd = np.zeros(shape=[batch_size, variable_dim])
    z01_bd = np.zeros(shape=[batch_size, variable_dim])

    s00_bd = np.zeros(shape=[batch_size, variable_dim])
    s01_bd = np.zeros(shape=[batch_size, variable_dim])

    t00_bd = np.zeros(shape=[batch_size, variable_dim])
    t01_bd = np.zeros(shape=[batch_size, variable_dim])

    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x00_bd[:, 0:1] = region_xleft
        x00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        x00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        x00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        x00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft

        x01_bd[:, 0:1] = region_xright
        x01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        x01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        x01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        x01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft

        y00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y00_bd[:, 1:2] = region_yleft
        y00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        y00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        y00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft

        y01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y01_bd[:, 1:2] = region_yright
        y01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        y01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        y01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft

        z00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        z00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z00_bd[:, 2:3] = region_zleft
        z00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        z00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft

        z01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        z01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z01_bd[:, 2:3] = region_zright
        z01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        z01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft

        s00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        s00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        s00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s00_bd[:, 3:4] = region_sleft
        s00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft

        s01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        s01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        s01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s01_bd[:, 3:4] = region_sright
        s01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft

        t00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        t00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        t00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        t00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        t00_bd[:, 4:5] = region_tleft

        t01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        t01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        t01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        t01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        t01_bd[:, 4:5] = region_tright
    else:
        x00_bd[:, 0:1] = region_xleft
        x00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        x00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        x00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        x00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft

        x01_bd[:, 0:1] = region_xright
        x01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        x01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        x01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        x01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft

        y00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        y00_bd[:, 1:2] = region_yleft
        y00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        y00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        y00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft

        y01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        y01_bd[:, 1:2] = region_yright
        y01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        y01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        y01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft

        z00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        z00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        z00_bd[:, 2:3] = region_zleft
        z00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        z00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft

        z01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        z01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        z01_bd[:, 2:3] = region_zright
        z01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        z01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft

        s00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        s00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        s00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        s00_bd[:, 3:4] = region_sleft
        s00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft

        s01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        s01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        s01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        s01_bd[:, 3:4] = region_sright
        s01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft

        t00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        t00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        t00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        t00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        t00_bd[:, 4:5] = region_tleft

        t01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        t01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        t01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        t01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        t01_bd[:, 4:5] = region_tright

    if to_float:
        x00_bd = x00_bd.astype(np.float32)
        x01_bd = x01_bd.astype(np.float32)

        y00_bd = y00_bd.astype(np.float32)
        y01_bd = y01_bd.astype(np.float32)

        z00_bd = z00_bd.astype(np.float32)
        z01_bd = z01_bd.astype(np.float32)

        s00_bd = s00_bd.astype(np.float32)
        s01_bd = s01_bd.astype(np.float32)

        t00_bd = t00_bd.astype(np.float32)
        t01_bd = t01_bd.astype(np.float32)

    if to_torch:
        x00_bd = torch.from_numpy(x00_bd)
        x01_bd = torch.from_numpy(x01_bd)

        y00_bd = torch.from_numpy(y00_bd)
        y01_bd = torch.from_numpy(y01_bd)

        z00_bd = torch.from_numpy(z00_bd)
        z01_bd = torch.from_numpy(z01_bd)

        s00_bd = torch.from_numpy(s00_bd)
        s01_bd = torch.from_numpy(s01_bd)

        t00_bd = torch.from_numpy(t00_bd)
        t01_bd = torch.from_numpy(t01_bd)

        if to_cuda:
            x00_bd = x00_bd.cuda(device='cuda:' + str(gpu_no))
            x01_bd = x01_bd.cuda(device='cuda:' + str(gpu_no))

            y00_bd = y00_bd.cuda(device='cuda:' + str(gpu_no))
            y01_bd = y01_bd.cuda(device='cuda:' + str(gpu_no))

            z00_bd = z00_bd.cuda(device='cuda:' + str(gpu_no))
            z01_bd = z01_bd.cuda(device='cuda:' + str(gpu_no))

            s00_bd = s00_bd.cuda(device='cuda:' + str(gpu_no))
            s01_bd = s01_bd.cuda(device='cuda:' + str(gpu_no))

            t00_bd = t00_bd.cuda(device='cuda:' + str(gpu_no))
            t01_bd = t01_bd.cuda(device='cuda:' + str(gpu_no))

        x00_bd.requires_grad = use_grad
        x01_bd.requires_grad = use_grad
        y00_bd.requires_grad = use_grad
        y01_bd.requires_grad = use_grad
        z00_bd.requires_grad = use_grad
        z01_bd.requires_grad = use_grad
        s00_bd.requires_grad = use_grad
        s01_bd.requires_grad = use_grad
        t00_bd.requires_grad = use_grad
        t01_bd.requires_grad = use_grad

    return x00_bd, x01_bd, y00_bd, y01_bd, z00_bd, z01_bd, s00_bd, s01_bd, t00_bd, t01_bd


def rand_bd_8D(batch_size=1000, variable_dim=8, region_xleft=0.0, region_xright=1.0, region_yleft=0.0, region_yright=1.0,
               region_zleft=0.0, region_zright=1.0, region_sleft=0.0, region_sright=1.0, region_tleft=0.0,
               region_tright=1.0, region_pleft=0.0, region_pright=1.0, region_qleft=0.0, region_qright=1.0,
               region_rleft=0.0, region_rright=1.0, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
               use_grad=False, opt2sampler='lhs'):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    assert variable_dim == 8
    x00_bd = np.zeros(shape=[batch_size, variable_dim])
    x01_bd = np.zeros(shape=[batch_size, variable_dim])

    y00_bd = np.zeros(shape=[batch_size, variable_dim])
    y01_bd = np.zeros(shape=[batch_size, variable_dim])

    z00_bd = np.zeros(shape=[batch_size, variable_dim])
    z01_bd = np.zeros(shape=[batch_size, variable_dim])

    s00_bd = np.zeros(shape=[batch_size, variable_dim])
    s01_bd = np.zeros(shape=[batch_size, variable_dim])

    t00_bd = np.zeros(shape=[batch_size, variable_dim])
    t01_bd = np.zeros(shape=[batch_size, variable_dim])

    p00_bd = np.zeros(shape=[batch_size, variable_dim])
    p01_bd = np.zeros(shape=[batch_size, variable_dim])

    q00_bd = np.zeros(shape=[batch_size, variable_dim])
    q01_bd = np.zeros(shape=[batch_size, variable_dim])

    r00_bd = np.zeros(shape=[batch_size, variable_dim])
    r01_bd = np.zeros(shape=[batch_size, variable_dim])

    if 'lhs' == opt2sampler:
        sampler = stqmc.LatinHypercube(d=1)
        x00_bd[:, 0:1] = region_xleft
        x00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        x00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        x00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        x00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        x00_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        x00_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        x00_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        x01_bd[:, 0:1] = region_xright
        x01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        x01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        x01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        x01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        x01_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        x01_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        x01_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        y00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y00_bd[:, 1:2] = region_yleft
        y00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        y00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        y00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        y00_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        y00_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        y00_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        y01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        y01_bd[:, 1:2] = region_yright
        y01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        y01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        y01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        y01_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        y01_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        y01_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        z00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        z00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z00_bd[:, 2:3] = region_zleft
        z00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        z00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        z00_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        z00_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        z00_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        z01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        z01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        z01_bd[:, 2:3] = region_zright
        z01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        z01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        z01_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        z01_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        z01_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        s00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        s00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        s00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s00_bd[:, 3:4] = region_sleft
        s00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        s00_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        s00_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        s00_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        s01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        s01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        s01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        s01_bd[:, 3:4] = region_sright
        s01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        z01_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        z01_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        z01_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        t00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        t00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        t00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        t00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        t00_bd[:, 4:5] = region_tleft
        t00_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        t00_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        t00_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        t01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        t01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        t01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        t01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        t01_bd[:, 4:5] = region_tright
        t01_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        t01_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        t01_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        p00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        p00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        p00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        p00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        p00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        p00_bd[:, 5:6] = region_pleft
        p00_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        p00_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        p01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        p01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        p01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        p01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        p01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        p01_bd[:, 5:6] = region_pright
        p01_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        p01_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        q00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        q00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        q00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        q00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        q00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        q00_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        q00_bd[:, 6:7] = region_qleft
        q00_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        q01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        q01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        q01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        q01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        q01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        q01_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        q01_bd[:, 6:7] = region_qright
        q01_bd[:, 7:8] = (region_rright - region_rleft) * sampler.random(batch_size) + region_rleft

        r00_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        r00_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        r00_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        r00_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        r00_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        r00_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        r00_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        r00_bd[:, 7:8] = region_rleft

        r01_bd[:, 0:1] = (region_xright - region_xleft) * sampler.random(batch_size) + region_xleft
        r01_bd[:, 1:2] = (region_yright - region_yleft) * sampler.random(batch_size) + region_yleft
        r01_bd[:, 2:3] = (region_zright - region_zleft) * sampler.random(batch_size) + region_zleft
        r01_bd[:, 3:4] = (region_sright - region_sleft) * sampler.random(batch_size) + region_sleft
        r01_bd[:, 4:5] = (region_tright - region_tleft) * sampler.random(batch_size) + region_tleft
        r01_bd[:, 5:6] = (region_pright - region_pleft) * sampler.random(batch_size) + region_pleft
        r01_bd[:, 6:7] = (region_qright - region_qleft) * sampler.random(batch_size) + region_qleft
        r01_bd[:, 7:8] = region_rright
    else:
        x00_bd[:, 0:1] = region_xleft
        x00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        x00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        x00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        x00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        x00_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        x00_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        x00_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        x01_bd[:, 0:1] = region_xright
        x01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        x01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        x01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        x01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        x01_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        x01_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        x01_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        y00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        y00_bd[:, 1:2] = region_yleft
        y00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        y00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        y00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        y00_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        y00_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        y00_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        y01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        y01_bd[:, 1:2] = region_yright
        y01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        y01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        y01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        y01_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        y01_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        y01_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        z00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        z00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        z00_bd[:, 2:3] = region_zleft
        z00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        z00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        z00_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        z00_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        z00_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        z01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        z01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        z01_bd[:, 2:3] = region_zright
        z01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        z01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        z01_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        z01_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        z01_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        s00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        s00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        s00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        s00_bd[:, 3:4] = region_sleft
        s00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        s00_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        s00_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        s00_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        s01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        s01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        s01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        s01_bd[:, 3:4] = region_sright
        s01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        z01_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        z01_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        z01_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        t00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        t00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        t00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        t00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        t00_bd[:, 4:5] = region_tleft
        t00_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        t00_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        t00_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        t01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        t01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        t01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        t01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        t01_bd[:, 4:5] = region_tright
        t01_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        t01_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        t01_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        p00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        p00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        p00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        p00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        p00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        p00_bd[:, 5:6] = region_pleft
        p00_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        p00_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        p01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        p01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        p01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        p01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        p01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        p01_bd[:, 5:6] = region_pright
        p01_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        p01_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        q00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        q00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        q00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        q00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        q00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        q00_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        q00_bd[:, 6:7] = region_qleft
        q00_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        q01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        q01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        q01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        q01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        q01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        q01_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        q01_bd[:, 6:7] = region_qright
        q01_bd[:, 7:8] = (region_rright - region_rleft) * np.random.random([batch_size, 1]) + region_rleft

        r00_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        r00_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        r00_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        r00_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        r00_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        r00_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        r00_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        r00_bd[:, 7:8] = region_rleft

        r01_bd[:, 0:1] = (region_xright - region_xleft) * np.random.random([batch_size, 1]) + region_xleft
        r01_bd[:, 1:2] = (region_yright - region_yleft) * np.random.random([batch_size, 1]) + region_yleft
        r01_bd[:, 2:3] = (region_zright - region_zleft) * np.random.random([batch_size, 1]) + region_zleft
        r01_bd[:, 3:4] = (region_sright - region_sleft) * np.random.random([batch_size, 1]) + region_sleft
        r01_bd[:, 4:5] = (region_tright - region_tleft) * np.random.random([batch_size, 1]) + region_tleft
        r01_bd[:, 5:6] = (region_pright - region_pleft) * np.random.random([batch_size, 1]) + region_pleft
        r01_bd[:, 6:7] = (region_qright - region_qleft) * np.random.random([batch_size, 1]) + region_qleft
        r01_bd[:, 7:8] = region_rright

    if to_float:
        x00_bd = x00_bd.astype(np.float32)
        x01_bd = x01_bd.astype(np.float32)

        y00_bd = y00_bd.astype(np.float32)
        y01_bd = y01_bd.astype(np.float32)

        z00_bd = z00_bd.astype(np.float32)
        z01_bd = z01_bd.astype(np.float32)

        s00_bd = s00_bd.astype(np.float32)
        s01_bd = s01_bd.astype(np.float32)

        t00_bd = t00_bd.astype(np.float32)
        t01_bd = t01_bd.astype(np.float32)

        p00_bd = t00_bd.astype(np.float32)
        p01_bd = t01_bd.astype(np.float32)

        q00_bd = t00_bd.astype(np.float32)
        q01_bd = t01_bd.astype(np.float32)

        r00_bd = t00_bd.astype(np.float32)
        r01_bd = t01_bd.astype(np.float32)

    if to_torch:
        x00_bd = torch.from_numpy(x00_bd)
        x01_bd = torch.from_numpy(x01_bd)

        y00_bd = torch.from_numpy(y00_bd)
        y01_bd = torch.from_numpy(y01_bd)

        z00_bd = torch.from_numpy(z00_bd)
        z01_bd = torch.from_numpy(z01_bd)

        s00_bd = torch.from_numpy(s00_bd)
        s01_bd = torch.from_numpy(s01_bd)

        t00_bd = torch.from_numpy(t00_bd)
        t01_bd = torch.from_numpy(t01_bd)

        p00_bd = torch.from_numpy(p00_bd)
        p01_bd = torch.from_numpy(p01_bd)

        q00_bd = torch.from_numpy(q00_bd)
        q01_bd = torch.from_numpy(q01_bd)

        r00_bd = torch.from_numpy(r00_bd)
        r01_bd = torch.from_numpy(r01_bd)

        if to_cuda:
            x00_bd = x00_bd.cuda(device='cuda:' + str(gpu_no))
            x01_bd = x01_bd.cuda(device='cuda:' + str(gpu_no))

            y00_bd = y00_bd.cuda(device='cuda:' + str(gpu_no))
            y01_bd = y01_bd.cuda(device='cuda:' + str(gpu_no))

            z00_bd = z00_bd.cuda(device='cuda:' + str(gpu_no))
            z01_bd = z01_bd.cuda(device='cuda:' + str(gpu_no))

            s00_bd = s00_bd.cuda(device='cuda:' + str(gpu_no))
            s01_bd = s01_bd.cuda(device='cuda:' + str(gpu_no))

            t00_bd = t00_bd.cuda(device='cuda:' + str(gpu_no))
            t01_bd = t01_bd.cuda(device='cuda:' + str(gpu_no))

            p00_bd = p00_bd.cuda(device='cuda:' + str(gpu_no))
            p01_bd = p01_bd.cuda(device='cuda:' + str(gpu_no))

            q00_bd = q00_bd.cuda(device='cuda:' + str(gpu_no))
            q01_bd = q01_bd.cuda(device='cuda:' + str(gpu_no))

            r00_bd = r00_bd.cuda(device='cuda:' + str(gpu_no))
            r01_bd = r01_bd.cuda(device='cuda:' + str(gpu_no))

        x00_bd.requires_grad = use_grad
        x01_bd.requires_grad = use_grad

        y00_bd.requires_grad = use_grad
        y01_bd.requires_grad = use_grad

        z00_bd.requires_grad = use_grad
        z01_bd.requires_grad = use_grad

        s00_bd.requires_grad = use_grad
        s01_bd.requires_grad = use_grad

        t00_bd.requires_grad = use_grad
        t01_bd.requires_grad = use_grad

        p00_bd.requires_grad = use_grad
        p01_bd.requires_grad = use_grad

        q00_bd.requires_grad = use_grad
        q01_bd.requires_grad = use_grad

        r00_bd.requires_grad = use_grad
        r01_bd.requires_grad = use_grad

    return x00_bd, x01_bd, y00_bd, y01_bd, z00_bd, z01_bd, s00_bd, s01_bd, t00_bd, t01_bd, \
           p00_bd, p01_bd, q00_bd, q01_bd, r00_bd, r01_bd


# the following codes added by Jiaxin Deng for ADE
def rand_bd_2D1(batch_size=100, variable_dim=2, region_left=0.1, region_right=1.0, region_bottom=0.0, region_top=1.0,
                to_torch=True, to_float=True, to_cuda=False, gpu_no=0, use_grad=False, lhs_sampling=True):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    # np.random.random((100, 50)) 上方代表生成100行 50列的随机浮点数，浮点数范围 : (0,1)
    # np.random.random([100, 50]) 和 np.random.random((100, 50)) 效果一样
    assert (int(variable_dim) == 2)
    if lhs_sampling:
        sampler = stqmc.LatinHypercube(d=variable_dim)
        x_left_bd = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom  # 浮点数都是从0-1中随机。
        x_right_bd = (region_top - region_bottom) * sampler.random(batch_size) + region_bottom # 浮点数都是从0-1中随机。
        y_bottom_bd = (region_right - region_left) * sampler.random(batch_size) + region_left
        y_top_bd = (region_right - region_left) * sampler.random(batch_size) + region_left
    else:
        y_bottom_bd = (region_right - region_left) * np.random.random([batch_size, 2]) + region_left  # 浮点数都是从0-1中随机。
        y_top_bd = (region_right - region_left) * np.random.random([batch_size, 2]) + region_left  # 浮点数都是从0-1中随机。
        x_left_bd = (region_top - region_bottom) * np.random.random([batch_size, 2]) + region_bottom
        x_right_bd = (region_top - region_bottom) * np.random.random([batch_size, 2]) + region_bottom

    # for ii in range(batch_size):
    #     x_left_bd[ii, 0] = region_left
    #     x_right_bd[ii, 0] = region_right
    #     y_bottom_bd[ii, 1] = region_bottom
    #     y_top_bd[ii, 1] = region_top

    x_left_bd[:, 0] = region_left
    x_right_bd[:, 0] = region_right
    y_bottom_bd[:, 1] = region_bottom
    y_top_bd[:, 1] = region_top

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


def rand_bd_2DV(batch_size, variable_dim, region_l, region_r, region_b, region_t,
                to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
                use_grad=False):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_l = float(region_l)
    region_r = float(region_r)
    region_b = float(region_b)
    region_t = float(region_t)

    assert (int(variable_dim) == 2)
    left_bd = np.random.rand(batch_size, 2)
    right_bd = np.random.rand(batch_size, 2)
    bottom_bd = np.random.rand(batch_size, 2)
    top_bd = np.random.rand(batch_size, 2)
    # 放缩过程
    left_bd = scale2D(left_bd, region_l, region_r, region_b, region_t)
    right_bd = scale2D(right_bd, region_l, region_r, region_b, region_t)
    bottom_bd = scale2D(bottom_bd, region_l, region_r, region_b, region_t)
    top_bd = scale2D(top_bd, region_l, region_r, region_b, region_t)

    left_bd[:, 0] = region_l
    right_bd[:, 0] = region_r
    bottom_bd[:, 1] = region_b
    top_bd[:, 1] = region_t

    if to_float:
        bottom_bd = bottom_bd.astype(np.float32)
        top_bd = top_bd.astype(np.float32)
        left_bd = left_bd.astype(np.float32)
        right_bd = right_bd.astype(np.float32)


    if to_torch:
        bottom_bd = torch.from_numpy(bottom_bd)
        top_bd = torch.from_numpy(top_bd)
        left_bd = torch.from_numpy(left_bd)
        right_bd = torch.from_numpy(right_bd)


        if to_cuda:
            bottom_bd = bottom_bd.cuda(device='cuda:' + str(gpu_no))
            top_bd = top_bd.cuda(device='cuda:' + str(gpu_no))
            left_bd = left_bd.cuda(device='cuda:' + str(gpu_no))
            right_bd = right_bd.cuda(device='cuda:' + str(gpu_no))


        bottom_bd.requires_grad = use_grad
        top_bd.requires_grad = use_grad
        left_bd.requires_grad = use_grad
        right_bd.requires_grad = use_grad

    return bottom_bd, top_bd, left_bd, right_bd


def scale2D(x, a, b, c, d):
    x[:, 0] = (b - a) * x[:, 0] + a
    x[:, 1] = (d - c) * x[:, 1] + c
    return x


def scale3D(x, a, b, c, d, e, f):
    x[:, 0] = (b - a) * x[:, 0] + a
    x[:, 1] = (d - c) * x[:, 1] + c
    x[:, 2] = (f - e) * x[:, 2] + e
    return x


def rand_bd_1DV_neu_hard(batch_size, variable_dim, region_a, region_b, init_l, init_r, to_torch=True, to_float=True,
                         to_cuda=False, gpu_no=0, use_grad=False, lhs_sampling=True):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    # np.random.random((100, 50)) 上方代表生成100行 50列的随机浮点数，浮点数范围 : (0,1)
    # np.random.random([100, 50]) 和 np.random.random((100, 50)) 效果一样
    region_a = float(region_a)
    region_b = float(region_b)
    assert (int(variable_dim) == 2)
    if lhs_sampling:
        sampler = stqmc.LatinHypercube(d=variable_dim)
        x_left_bd = (init_r - init_l) * sampler.random(batch_size) + init_l  # 浮点数都是从0-1中随机。
        x_right_bd = (init_r - init_l) * sampler.random(batch_size) + init_l  # 浮点数都是从0-1中随机。
        y_bottom_bd = (region_b - region_a) * sampler.random(batch_size) + region_a
        y_top_bd = (region_b - region_a) * sampler.random(batch_size) + region_a
    else:
        x_left_bd = (init_r - init_l) * np.random.random([batch_size, 2]) + init_l  # 浮点数都是从0-1中随机。
        x_right_bd = (init_r - init_l) * np.random.random([batch_size, 2]) + init_l  # 浮点数都是从0-1中随机。
        y_bottom_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
        y_top_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
    for ii in range(batch_size):
        x_left_bd[ii, 0] = init_l
        x_right_bd[ii, 0] = init_r
        y_bottom_bd[ii, 1] = region_a
        y_top_bd[ii, 1] = region_b

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


def rand2D_2(batch_size, variable_dim, region_l, region_r, region_b, region_t,
             to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
             use_grad=False, lhs_sampling=True):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_l = float(region_l)
    region_r = float(region_r)
    region_b = float(region_b)
    region_t = float(region_t)

    assert (int(variable_dim) == 2)
    if lhs_sampling:
        sampler = stqmc.LatinHypercube(d=variable_dim)
        xy_it = sampler.random(batch_size)
        xy_it[:, 0] = (region_r - region_l) * xy_it[:, 0] + region_l
        xy_it[:, 1] = (region_t - region_b) * xy_it[:, 1] + region_b
    else:
        xy_it = np.random.rand(batch_size, 2)
        xy_it[:, 0] = (region_r - region_l) * xy_it[:, 0] + region_l
        xy_it[:, 1] = (region_t - region_b) * xy_it[:, 1] + region_b

    if to_float:
        xy_it = xy_it.astype(np.float32)

    if to_torch:
        xy_it = torch.from_numpy(xy_it)

        if to_cuda:
            xy_it = xy_it.cuda(device='cuda:' + str(gpu_no))
        if use_grad:
                xy_it.requires_grad = use_grad
    return xy_it


def rand_bd_3DV(batch_size, variable_dim, region_l, region_r, region_b, region_t, region_f, region_be,
                to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
                use_grad=False):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_l = float(region_l)
    region_r = float(region_r)
    region_b = float(region_b)
    region_t = float(region_t)
    region_f = float(region_f)
    region_be = float(region_be)

    assert (int(variable_dim) == 3)
    sampler = stqmc.LatinHypercube(d=variable_dim)

    left_bd = sampler.random(batch_size)
    right_bd = sampler.random(batch_size)
    bottom_bd = sampler.random(batch_size)
    top_bd = sampler.random(batch_size)
    front_bd = sampler.random(batch_size)
    behind_bd = sampler.random(batch_size)

    # 放缩过程
    left_bd = scale3D(left_bd, region_l, region_r, region_b, region_t, region_f, region_be)
    right_bd = scale3D(right_bd, region_l, region_r, region_b, region_t, region_f, region_be)
    bottom_bd = scale3D(bottom_bd, region_l, region_r, region_b, region_t, region_f, region_be)
    top_bd = scale3D(top_bd, region_l, region_r, region_b, region_t, region_f, region_be)
    front_bd = scale3D(top_bd, region_l, region_r, region_b, region_t, region_f, region_be)
    behind_bd = scale3D(top_bd, region_l, region_r, region_b, region_t, region_f, region_be)

    left_bd[:, 0] = region_l
    right_bd[:, 0] = region_r
    bottom_bd[:, 1] = region_b
    top_bd[:, 1] = region_t
    front_bd[:, 2] = region_f
    behind_bd[:, 2] = region_be

    if to_float:
        bottom_bd = bottom_bd.astype(np.float32)
        top_bd = top_bd.astype(np.float32)
        left_bd = left_bd.astype(np.float32)
        right_bd = right_bd.astype(np.float32)
        front_bd = front_bd.astype(np.float32)
        behind_bd = behind_bd.astype(np.float32)

    if to_torch:
        bottom_bd = torch.from_numpy(bottom_bd)
        top_bd = torch.from_numpy(top_bd)
        left_bd = torch.from_numpy(left_bd)
        right_bd = torch.from_numpy(right_bd)
        front_bd = torch.from_numpy(front_bd)
        behind_bd = torch.from_numpy(behind_bd)
        if to_cuda:
            bottom_bd = bottom_bd.cuda(device='cuda:' + str(gpu_no))
            top_bd = top_bd.cuda(device='cuda:' + str(gpu_no))
            left_bd = left_bd.cuda(device='cuda:' + str(gpu_no))
            right_bd = right_bd.cuda(device='cuda:' + str(gpu_no))
            front_bd = front_bd.cuda(device='cuda:' + str(gpu_no))
            behind_bd = behind_bd.cuda(device='cuda:' + str(gpu_no))

        bottom_bd.requires_grad = use_grad
        top_bd.requires_grad = use_grad
        left_bd.requires_grad = use_grad
        right_bd.requires_grad = use_grad
        front_bd.requires_grad = use_grad
        behind_bd.requires_grad = use_grad

    return bottom_bd, top_bd, left_bd, right_bd, front_bd, behind_bd


def rand_it_seq(batch_size, variable_dim, region_a, region_b, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
                use_grad2x=False, lhs_sampling=True,i_epoch=None,Max_iter=None):
        # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
        # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
        x_it = np.zeros(variable_dim, batch_size)
        if lhs_sampling:
            sampler = stqmc.LatinHypercube(d=variable_dim)
            x_it[0:batch_size/2] = (region_b - region_a) * sampler.random(batch_size//2) + region_a
            x_it[batch_size/2:batch_size-1] = (region_b - region_a) * sampler.random(batch_size//2) + region_a

        else:
            x_it = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
        x_it = np.reshape(x_it, [batch_size, variable_dim])
        if to_float:
            x_it = x_it.astype(np.float32)

        if to_torch:
            x_it = torch.from_numpy(x_it)

            if to_cuda:
                x_it = x_it.cuda(device='cuda:' + str(gpu_no))

            x_it.requires_grad = use_grad2x

        return x_it


def rand_it_zeromore(batch_size, variable_dim, region_a, region_b, to_torch=True, to_float=True, to_cuda=False, gpu_no=0,
            use_grad2x=False, lhs_sampling=True, portion=0.3, split=0.1):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    if lhs_sampling:
        sampler = stqmc.LatinHypercube(d=variable_dim)
        x_it_1 = (region_b-region_a) * sampler.random(int(batch_size*(1-portion))) + region_a + split * (region_b-region_a)
        x_it_2 = sampler.random(batch_size-int(batch_size*(1-portion))) * split * (region_b-region_a) + region_a
        x_it = np.concatenate([x_it_1, x_it_2], axis=0)
    else:
        x_it = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    # x_it = np.reshape(x_it, [batch_size, variable_dim])
    if to_float:
        x_it = x_it.astype(np.float32)

    if to_torch:
        x_it = torch.from_numpy(x_it)

        if to_cuda:
            x_it = x_it.cuda(device='cuda:' + str(gpu_no))

        x_it.requires_grad = use_grad2x

    return x_it


if __name__ == "__main__":
    x_it = rand_it_zeromore(batch_size=3000, variable_dim=1, region_a=0, region_b=10, to_torch=False,
                            to_float=True, to_cuda=False, gpu_no=0, use_grad2x=False, lhs_sampling=True)