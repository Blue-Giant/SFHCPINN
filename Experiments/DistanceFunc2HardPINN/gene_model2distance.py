import numpy as np
import torch
from Networks.NETWORK import *
from Utilizers.gen_data import *
import matplotlib.pyplot as plt
import os


def compute_dist(x_in=None, x_bd=None, dim=2, normalize=False):
    # xs: List of collocation points
    # xbs: List of boundary conditions
    if normalize:
        max_xin = np.reshape(np.max(x_in, axis=0), newshape=(1, -1))
        x_in = np.divide(x_in, max_xin)

        max_xbd = np.reshape(np.max(x_bd, axis=0), newshape=(1, -1))
        x_bd = np.divide(x_bd, max_xbd)

    if 2 == dim:
        xs1 = np.reshape(x_in[:, 0], newshape=[-1, 1])
        xb1 = np.reshape(x_bd[:, 0], newshape=[1, -1])

        xs2 = np.reshape(x_in[:, 1], newshape=[-1, 1])
        xb2 = np.reshape(x_bd[:, 1], newshape=[1, -1])

        diff1 = xs1 - xb1
        diff2 = xs2 - xb2

        norm2diff = np.sqrt(np.square(diff1) + np.square(diff2))
    else:
        xs1 = np.reshape(x_in[:, 0], newshape=[-1, 1])
        xb1 = np.reshape(x_bd[:, 0], newshape=[1, -1])

        xs2 = np.reshape(x_in[:, 1], newshape=[-1, 1])
        xb2 = np.reshape(x_bd[:, 1], newshape=[1, -1])

        xs3 = np.reshape(x_in[:, 2], newshape=[-1, 1])
        xb3 = np.reshape(x_bd[:, 2], newshape=[1, -1])

        diff1 = xs1 - xb1
        diff2 = xs2 - xb2
        diff3 = xs3 - xb3

        norm2diff = np.sqrt(np.square(diff1) + np.square(diff2) + np.square(diff3))

    ds = np.min(norm2diff, axis=-1, keepdims=True)

    # ds = [min([np.linalg.norm(x - xb) for xb in xbs]) for x in xs]
    return ds


def gene_in_bd_points_dist2double_dirichlet(point_num2inside=1000, point_num2inflowbd=100, region_a=0.0,
                                            region_b=1.0, time_begin=0.1, time_end=1.0, variable_dim=2):
    # 随机生成区域内部点
    random_points2in = rand_bd_inside(point_num2inside, variable_dim, region_a, region_b, time_begin, time_end,
                                      to_torch=False, to_float=True, opt2sampler='lhs')

    # # 迪利克雷边界: 随机生成区域边界点（空间左右边界，时间初始边界）---------- 初始条件可以看作迪利克雷边界
    random_points2bd = rand_bd_inflow2double_dirichlet(
        point_num2inflowbd, variable_dim, region_a, region_b, time_begin, time_end, to_torch=False, to_float=True,
        opt2sampler='lhs')
    # Random set to train of train_size random points in the domain (excluding extremal points)
    distance2in_bd = compute_dist(x_in=random_points2in, x_bd=random_points2bd, dim=2, normalize=False)  # 内部点和边界点的距离
    x_train, y_train = tensorv(random_points2in), tensorv(distance2in_bd)             # y_train 是内部点和inflow边界点的距离 要使这个最小
    return x_train, y_train


def gene_in_bd_points_dist2double_neumann(point_num2inside=1000, point_num2inflowbd=100, region_a=0.0,
                                          region_b=1.0, time_begin=0.1, time_end=1.0, variable_dim=2):
    # 随机生成区域内部点
    random_points2in = rand_bd_inside(point_num2inside, variable_dim, region_a, region_b, time_begin, time_end,
                                      to_torch=False, to_float=True, opt2sampler='lhs')

    # 随机生成区域边界点。处理左右边界为诺伊曼边界问题,时间结束边界不存在,那么只有初始边界可以使用
    random_points2bd = rand_inflow2double_neumann_bd(point_num2inflowbd, variable_dim, region_a, region_b, time_begin,
                                                     time_end, to_torch=False, to_float=True, opt2sampler='lhs')

    # Random set to train of train_size random points in the domain (excluding extremal points)
    distance2in_bd = compute_dist(x_in=random_points2in, x_bd=random_points2bd, dim=2, normalize=False)  # 内部点和边界点的距离
    x_train, y_train = tensorv(random_points2in), tensorv(distance2in_bd)             # y_train 是内部点和inflow边界点的距离 要使这个最小
    return x_train, y_train


def Gene_Distance_Model2Dirichlet(epochs=1000, num2in_point=100, num_inflowbd_points=50, region_l=0.0, region_r=1.0,
                                  init_begin=0.0, init_end=1.0, name2model='DNN', width2nn=50, actIn_Name='tanh',
                                  actHidden_Name='tanh', actOut_Name='linear', init_learning_rate=0.01,
                                  with_gpu=False, no2gpu=0, gamma2stepLR=0.95):
    """
    Args:
        epochs: 训练轮数
        num2in_point: 内部点数目
        num_inflowbd_points: 边界流数目
        region_l: 空间变量左边界
        region_r: 空间变量右边界
        init_begin: 时间变量初始时刻
        init_end: 时间变量结束时刻
        init_learning_rate:
        with_gpu:
        no2gpu:
        gamma2stepLR:
    Returns:
        model2nn
    """

    if name2model == 'Fourier_DNN':
        model_distance = Fourier_FCN(indim=2, outdim=1, width=width2nn, actName2In=actIn_Name,
                                     actName2Hidden=actHidden_Name, actName2Out=actOut_Name, type2float='float32',
                                     to_gpu=with_gpu, gpu_no=no2gpu)
    else:
        model_distance = FCN(indim=2, outdim=1, width=width2nn, actName2In=actIn_Name, actName2Hidden=actHidden_Name,
                             actName2Out=actOut_Name, type2float='float32', to_gpu=with_gpu, gpu_no=no2gpu)
    params2Net = model_distance.parameters()
    optimizer = torch.optim.Adam(params2Net, lr=init_learning_rate)  # Adam
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=gamma2stepLR)
    print('---------------------- start training Model of distance --------------------------')
    for epoch in range(epochs+1):
        x_train, y_train = gene_in_bd_points_dist2double_dirichlet(
            point_num2inside=num2in_point, point_num2inflowbd=num_inflowbd_points, region_a=region_l,
            region_b=region_r, time_begin=init_begin, time_end=init_end)

        if True == with_gpu:
            x_train = x_train.cuda(device='cuda:' + str(no2gpu))
            y_train = y_train.cuda(device='cuda:' + str(no2gpu))

        y_pre = model_distance(x_train)
        loss = torch.mean(torch.square(y_train-y_pre))

        optimizer.zero_grad()  # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()        # 对loss关于Ws和Bs求偏导
        optimizer.step()       # 更新参数Ws和Bs
        scheduler.step()

        if epoch % 100 == 0:
            tmp_lr = optimizer.param_groups[0]['lr']
            print('epochs:%d  ----lr:%.8f  -----mse: %.15f' % (epoch, tmp_lr, loss.item()))
    return model_distance


def Gene_Distance_Model2Neumann(epochs=1000, num2in_point=100, num_inflowbd_points=50, region_l=0.0, region_r=1.0,
                        init_begin=0.0, init_end=1.0, name2model='DNN', width2nn=50, actIn_Name='tanh',
                        actHidden_Name='tanh', actOut_Name='linear', init_learning_rate=0.01,
                        with_gpu=False, no2gpu=0, gamma2stepLR=0.95):

    if name2model == 'DNN':
        model_distance = FCN(indim=2, outdim=1, width=width2nn, actName2In=actIn_Name, actName2Hidden=actHidden_Name,
                             actName2Out=actOut_Name, type2float='float32', to_gpu=with_gpu, gpu_no=no2gpu)
    else:
        model_distance = Fourier_FCN(indim=2, outdim=1, width=width2nn, actName2In=actIn_Name, actName2Hidden=actHidden_Name,
                                     actName2Out=actOut_Name, type2float='float32', to_gpu=with_gpu, gpu_no=no2gpu)
    params2Net = model_distance.parameters()
    optimizer = torch.optim.Adam(params2Net, lr=init_learning_rate)  # Adam
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=gamma2stepLR)

    print('---------start training Model D for Neumann boundary--------')
    for epoch in range(epochs):
        x_train, y_train = gene_in_bd_points_dist2double_neumann(
            point_num2inside=num2in_point, point_num2inflowbd=num_inflowbd_points, region_a=region_l,
            region_b=region_r, time_begin=init_begin, time_end=init_end, variable_dim=2)
        if True == with_gpu:
            x_train = x_train.cuda(device='cuda:' + str(no2gpu))
            y_train = y_train.cuda(device='cuda:' + str(no2gpu))
        y_pre = model_distance(x_train)
        loss = torch.mean(torch.square(y_train-y_pre))

        optimizer.zero_grad()  # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()        # 对loss关于Ws和Bs求偏导
        optimizer.step()       # 更新参数Ws和Bs
        scheduler.step()

        if epoch % 100 == 0:
            tmp_lr = optimizer.param_groups[0]['lr']
            print('epochs:%d  ----lr:%.8f  -----mse: %.15f' % (epoch, tmp_lr, loss.item()))
    return model_distance


def plot_distance_function(model_distance, region_a, region_b, init_l, init_r, splits):
    y, x = np.meshgrid(np.linspace(init_l, init_r, splits), np.linspace(region_a, region_b, splits))
    points = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], 1)
    points = tensorv(points)
    z = model_distance(points).detach().numpy()
    a = z.reshape(splits, splits)
    z_min, z_max = np.nanmin(z), np.nanmax(z)
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_title('model_distance function')
    c = ax.pcolormesh(x, y, a, cmap='RdBu', vmin=z_min, vmax=z_max)
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.show()
    return z, points


# -------------------------------------------------------------------------------------------------------------------
def gen_xy_distance1(num_inside_points, num_inflowbd_points, region_a, region_b, init_l, init_r, variable_dim=2):
    random_inside_points = rand_bd_inside(num_inside_points, variable_dim, region_a, region_b, init_l, init_r,
                                          to_torch=False, to_float=True)
    x_train = np.array(random_inside_points).reshape(-1, 2)  # random_inside points是内部随机点
    random_inside_points = scaling(random_inside_points, region_a, region_b, init_l, init_r).reshape(-1, 2)

    random_boundary_points = rand_bd_inflow(num_inflowbd_points, variable_dim, region_a, region_b, init_l, init_r,
                                            to_torch=False, to_float=True)
    random_boundary_points = scaling(random_boundary_points, region_a, region_b, init_l, init_r)
    # Random set to train of train_size random points in the domain (excluding extremal points)

    ds = compute_dist(x_in=random_inside_points, x_bd=random_boundary_points)  # 内部点和inflow边界点的距离
    y_train = np.array(ds).reshape(-1, 1)  # y_train 是内部点和inflow边界点的距离 要使这个最小
    x_train, y_train = tensorv(x_train), tensorv(y_train)
    return x_train, y_train


def scaling(x_in_batch, region_l, region_r, init_time, end_time):
    x_in_batch[:, 0] = (x_in_batch[:, 0] - region_l) / (region_r - region_l)
    x_in_batch[:, 1] = (x_in_batch[:, 1] - init_time) / (end_time - init_time)
    return x_in_batch


def Gen_model_distance2(epochs=1000, num2in_point=100, num_inflowbd_points=50, region_l=0.0, region_r=1.0,
                        init_begin=0.0, init_end=1.0, name2model='DNN', width2nn=50, actIn_Name='tanh',
                        actHidden_Name='tanh', actOut_Name='linear', init_learning_rate=0.01,
                        with_gpu=False, no2gpu=0, gamma2stepLR=0.95):
    if name2model != 'DNN':
        model_distance = Fourier_FCN(indim=2, outdim=1, width=width2nn, actName2In=actIn_Name,
                                     actName2Hidden=actHidden_Name,
                                     actName2Out=actOut_Name, type2float='float32', to_gpu=with_gpu, gpu_no=no2gpu)
    else:
        model_distance = FCN(indim=2, outdim=1, width=width2nn, actName2In=actIn_Name, actName2Hidden=actHidden_Name,
                             actName2Out=actOut_Name, type2float='float32', to_gpu=with_gpu, gpu_no=no2gpu)

    params2Net = model_distance.parameters()
    optimizer = torch.optim.Adam(params2Net, lr=init_learning_rate)  # Adam
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=gamma2stepLR)
    print('---------start training Model D')
    for epoch in range(epochs):
        x_train, y_train = gen_xy_distance2(num2in_point, num_inflowbd_points, region_l, region_r, init_begin, init_end)
        y_pre = model_distance(x_train)
        loss = torch.mean(torch.square(y_train - y_pre))

        optimizer.zero_grad()  # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()  # 对loss关于Ws和Bs求偏导
        optimizer.step()  # 更新参数Ws和Bs
        scheduler.step()

        if epoch % 100 == 0:
            tmp_lr = optimizer.param_groups[0]['lr']
            print('epochs:%d  ----lr:%.8f  -----mse: %.15f' % (epoch, tmp_lr, loss.item()))
    return model_distance


# 生成 N个 边界点以及 M个 流入点（边界控制点）
def gen_xy_distance2(num_inside_points, num_inflowbd_points, region_a, region_b, init_l, init_r, variable_dim=2):
    random_boundary_points = rand_bd_inflow(num_inflowbd_points, variable_dim, region_a, region_b, init_l, init_r,
                                            to_torch=False, to_float=True)
    random_inside_points = rand_bd_inside(num_inside_points, variable_dim, region_a, region_b, init_l, init_r,
                                          to_torch=False, to_float=True)
    x_train = np.concatenate([random_inside_points, random_boundary_points], 0).reshape(-1,
                                                                                        2)  # random_inside points是内部随机点
    random_inside_points = scaling(random_inside_points, region_a, region_b, init_l, init_r).reshape(-1, 2)

    random_boundary_points = scaling(random_boundary_points, region_a, region_b, init_l, init_r)
    random_points = np.concatenate([random_inside_points, random_boundary_points], 0).reshape(-1, 2)
    # Random set to train of train_size random points in the domain (excluding extremal points)
    ds = compute_dist(random_points, random_boundary_points)  # 内部点和inflow边界点的距离
    y_train = np.array(ds).reshape(-1, 1)  # y_train 是内部点和inflow边界点的距离 要使这个最小
    x_train, y_train = tensorv(x_train), tensorv(y_train)
    return x_train, y_train


def Gen_model_distance_EX6(epochs=1000, num2in_point=100, num_inflowbd_points=50, region_l=0.0, region_r=1.0,
                        init_begin=0.0, init_end=1.0, name2model='DNN', width2nn=50, actIn_Name='tanh',
                        actHidden_Name='tanh', actOut_Name='linear', init_learning_rate=0.01,
                        with_gpu=False, no2gpu=0, gamma2stepLR=0.95, SCALE=False):
    if name2model != 'DNN':
        model_distance = Fourier_FCN(indim=2, outdim=1, width=width2nn, actName2In=actIn_Name,
                                     actName2Hidden=actHidden_Name,
                                     actName2Out=actOut_Name, type2float='float32', to_gpu=with_gpu, gpu_no=no2gpu)
    else:
        model_distance = FCN(indim=2, outdim=1, width=width2nn, actName2In=actIn_Name, actName2Hidden=actHidden_Name,
                             actName2Out=actOut_Name, type2float='float32', to_gpu=with_gpu, gpu_no=no2gpu)

    params2Net = model_distance.parameters()
    optimizer = torch.optim.Adam(params2Net, lr=init_learning_rate)  # Adam
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=gamma2stepLR)
    print('---------start training Model D')
    for epoch in range(epochs):
        x_train, y_train = gen_xy_distance_EX6(num2in_point, num_inflowbd_points, region_l, region_r, init_begin,
                                               init_end, norm=SCALE)
        y_pre = model_distance(x_train)
        loss = torch.mean(torch.square(y_train - y_pre))

        optimizer.zero_grad()  # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()  # 对loss关于Ws和Bs求偏导
        optimizer.step()  # 更新参数Ws和Bs
        scheduler.step()

        if epoch % 100 == 0:
            tmp_lr = optimizer.param_groups[0]['lr']
            print('epochs:%d  ----lr:%.8f  -----mse: %.15f' % (epoch, tmp_lr, loss.item()))
    return model_distance


# 生成 N个 边界点以及 M个 流入点（边界控制点）
def gen_xy_distance_EX6(num_inside_points, num_inflowbd_points, region_a, region_b, init_l, init_r, norm=False):
    if norm:
        random_inside_points = rand_bd_inside_EX6(num_inside_points, region_a, region_b, init_l, init_r,
                                                  to_torch=False, to_float=True)
        random_boundary_points = rand_bd_inflow_EX6(num_inflowbd_points, region_a, region_b, init_l, init_r,
                                                    to_torch=False, to_float=True)
        x_train = np.concatenate([random_boundary_points, random_inside_points], 0).reshape(-1,
                                                                                            2)  # random_inside points是内部随机点
        # 放缩
        random_inside_points = scaling(random_inside_points, region_a, region_b, init_l, init_r)
        random_boundary_points = scaling(random_boundary_points, region_a, region_b, init_l, init_r)
        inputs = np.concatenate([random_boundary_points, random_inside_points], 0).reshape(-1, 2)
        # 生成输出
        ds = compute_dist(inputs, random_boundary_points)  # 内部点和inflow边界点的距离

    else:
        random_inside_points = rand_bd_inside_EX6(num_inside_points, region_a, region_b, init_l, init_r,
                                                  to_torch=False, to_float=True)
        random_boundary_points = rand_bd_inflow_EX6(num_inflowbd_points, region_a, region_b, init_l, init_r,
                                                    to_torch=False, to_float=True)
        # Random set to train of train_size random points in the domain (excluding extremal points)
        x_train = np.concatenate([random_boundary_points, random_inside_points], 0).reshape(-1, 2)
        # random_inside points是内部随机点 加randon_boundary_points是为了满足&加深inflow边界点为0这个边界

        ds = compute_dist(x_train, random_boundary_points)  # 内部点和inflow边界点的距离

    y_train = np.array(ds).reshape(-1, 1)  # y_train 是内部点和inflow边界点的距离 要使这个最小
    x_train, y_train = tensorv(x_train), tensorv(y_train)
    return x_train, y_train


if __name__ == "__main__":
    equa_name ='General'
    # equa_name = 'Multiscale'

    # type2bd = 'Dirichlet'
    type2bd = 'Neumann'

    if equa_name == 'General':
        region_a = 0.0
        region_b = 2.0
        init_time = 0.0
        end_time = 5.0
    else:
        region_a = 0.0
        region_b = 1.0
        init_time = 0.0
        end_time = 1.0

    # model2nn = 'DNN'
    model2nn = 'Fourier_DNN'

    if type2bd == 'Dirichlet':
        model_DD = Gene_Distance_Model2Dirichlet(
            epochs=20000, num2in_point=2500, num_inflowbd_points=250,
            region_l=region_a, region_r=region_b, init_begin=init_time, init_end=end_time,
            name2model=model2nn, width2nn=50, actIn_Name='tanh', actHidden_Name='tanh',
            actOut_Name='linear', init_learning_rate=0.01, with_gpu=True, no2gpu=0,
            gamma2stepLR=0.95)
        if equa_name == 'General':
            torch.save(model_DD, "Dirichlet1D_DistModel2Space02Time05.pth")  # 保存整个模型
        else:
            torch.save(model_DD, "Dirichlet1D_DistModel2Space01Time01.pth")  # 保存整个模型
    else:
        model_DN = Gene_Distance_Model2Neumann(
            epochs=20000, num2in_point=2500, num_inflowbd_points=250,
            region_l=region_a, region_r=region_b, init_begin=init_time, init_end=end_time,
            name2model=model2nn, width2nn=50, actIn_Name='tanh', actHidden_Name='tanh',
            actOut_Name='linear', init_learning_rate=0.01, with_gpu=True, no2gpu=0,
            gamma2stepLR=0.95)
        if equa_name == 'General':
            torch.save(model_DN, "Neumann1D_DistModel2Space02Time05.pth")  # 保存整个模型
        else:
            torch.save(model_DN, "Neumann1D_DistModel2Space01Time01.pth")  # 保存整个模型