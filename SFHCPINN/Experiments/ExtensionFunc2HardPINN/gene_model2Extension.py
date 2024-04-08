from torch import nn
import torch.optim as optim
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import sys


def rand_bd_ex1(batch_size, region_a, region_b, init_l, init_r, to_torch=True, to_float=True,
                to_cuda=False, gpu_no=0, use_grad=True):
    '''

    Args:
        batch_size: 生成的内部点个数
        region_a：左边界
        region_b: 右边界
        init_l: 初始值
        init_r: 结束时间
        to_torch:
        to_float:
        to_cuda:
        gpu_no:
        use_grad:

    Returns:

    '''
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


def rand_inflow(batch_size, region_a, region_b, init_l, init_r, to_torch=True, to_float=True,
                to_cuda=False, gpu_no=0, use_grad=True):
    '''
    这个函数主要用于生成Example1 的边界
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
    region_a = float(region_a)
    region_b = float(region_b)
    init_l = float(init_l)
    init_r = float(init_r)
    inflow_points = np.random.random([batch_size, 2])
    inflow_points[0:int(batch_size / 4), 1] = init_l
    inflow_points[int(batch_size / 4):int(batch_size / 2), 0] = region_a
    inflow_points[int(batch_size / 2):batch_size, 0] = region_b

    if to_float:
        inflow_points = inflow_points.astype(np.float32)

    if to_torch:
        inflow_points = torch.from_numpy(inflow_points)
        if to_cuda:
            inflow_points = inflow_points.cuda(device='cuda:' + str(gpu_no))
        inflow_points.requires_grad = use_grad
    return inflow_points

def tensorv(x):
    x = torch.FloatTensor(x)
    #     return Variable(x,requires_grad=True)
    return Variable(x, requires_grad=True)


def d(inside_points, inflow_points, norm=True):
    '''
    计算内部点到边界点的最短距离
    Args:
        xs: 内部点集合
        xbs: 边界点集合

    Returns:

    '''
    # xs: List of collocation points
    # xbs: List of boundary conditions
    if norm:
        inside_points, inflow_points = np.array(inside_points), np.array(inflow_points)
        max_0 = max(inside_points[:, 0])
        max_1 = max(inside_points[:, 1])
        inside_points[:, 0] = inside_points[:, 0] / max_0
        inside_points[:, 1] = inside_points[:, 1] / max_1
        inflow_points[:, 0] = inflow_points[:, 0] / max_0
        inflow_points[:, 1] = inflow_points[:, 1] / max_1
        ds = [min([np.linalg.norm(x - xb) for xb in inflow_points]) for x in inside_points]
    else:
        inside_points, inflow_points = np.array(inside_points), np.array(inflow_points)
        ds = [min([np.linalg.norm(x - xb) for xb in inflow_points]) for x in inside_points]
    return ds


def Gen_model_distance(epochs, num_inside_points, num_inflowbd_points, region_a, region_b, init_l, init_r):
    model_distance = Network()
    print('---------start training Model D--------')
    for epoch in range(epochs):
        model_distance.optimizer.zero_grad()
        inside_points = rand_bd_ex1(batch_size=num_inside_points, region_a=region_a, region_b=region_b, init_l=init_l,
                                    init_r=init_r, to_torch=False, to_float=True,
                                    to_cuda=False, gpu_no=0, use_grad=True)
        inflow_points = rand_inflow(batch_size=num_inflowbd_points, region_a=region_a, region_b=region_b, init_l=init_l,
                                    init_r=init_r, to_torch=False, to_float=True,
                                    to_cuda=False, gpu_no=0, use_grad=True)
        label_norm = d(inside_points, inflow_points, norm=True)
        x_train, y_train = tensorv(inside_points), tensorv(label_norm)
        x_train = x_train.cuda(device='cuda:' + str(gpu_no))
        y_train = y_train.cuda(device='cuda:' + str(gpu_no))
        y_train = y_train.reshape(-1, 1)
        pred = model_distance(x_train)
        loss = model_distance.criterion(pred, y_train)
        loss.backward()
        model_distance.optimizer.step()
        run_loss = loss.item()
        num = 100
        if epoch % num == num - 1:
            print('epochs:%d   MSEloss : %.8f' % (epoch + 1, run_loss / num))
    return model_distance


def plot_distance_function(model_distance, region_a, region_b, init_l, init_r, splits):
    seed = np.random.randint(1e5)
    seed_str = str(seed)  # int 型转为字符串型
    y, x = np.meshgrid(np.linspace(init_l, init_r, splits), np.linspace(region_a, region_b, splits))
    points = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], 1)
    points = tensorv(points)
    # points = points.cuda(device='cuda:' + str(0))
    z = model_distance(points).detach().numpy()
    a = z.reshape(splits, splits)
    z_min, z_max = np.nanmin(z), np.nanmax(z)
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_title('model_distance function')
    c = ax.pcolormesh(x, y, a, cmap='RdBu', shading='auto', vmin=z_min, vmax=z_max)
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)

    plt.savefig("distance" + seed_str + ".jpg")
    plt.show()
    return z, points


# def Gen_model_G(epochs, num_inflowbd_points, region_a, region_b, init_l, init_r, u_true):
#     model_G = Network()
#     model_G.cuda()
#     print('---------start training Model G')
#     for epoch in range(epochs):
#         model_G.optimizer.zero_grad()
#         inflow_points = rand_inflow(batch_size=num_inflowbd_points, region_a=region_a, region_b=region_b, init_l=init_l,
#                                     init_r=init_r, to_torch=True, to_float=True,
#                                     to_cuda=True, gpu_no=0, use_grad=True)
#         x_train = inflow_points  # 用边界点作为输入
#         y_train = u_true(inflow_points[:, 0], inflow_points[:, 1]).reshape(-1, 1)
#         pred = model_G(x_train)
#         loss = model_G.criterion(pred, y_train)
#         loss.backward()
#         model_G.optimizer.step()
#         run_loss = loss.item()
#         num = 100
#         if epoch % num == num - 1:
#             print('epochs:%d   MSEloss : %.9f' % (epoch + 1, run_loss / num))
#     return model_G


# def Gen_model_G(epochs, num_inflowbd_points, region_a, region_b, init_l, init_r, g):
#     model_G = Network()
#     print('---------start training Model G')
#     for epoch in range(epochs):
#         model_G.optimizer.zero_grad()
#         inflow_boundary_points = rand_bd_inflow(num_inflowbd_points, 2, region_a, region_b, init_l, init_r,
#                                                 to_torch=True, to_float=True)
#         x_train = inflow_boundary_points  # 用边界点作为输入
#         y_train = g(inflow_boundary_points[:, 0], inflow_boundary_points[:, 1])  # g(x,y)作为输出
#         x_train, y_train = tensorv(x_train), tensorv(y_train)
#         pred = model_G(x_train)
#         loss = model_G.criterion(pred, y_train)
#         loss.backward()
#         model_G.optimizer.step()
#         run_loss = loss.item()
#         num = 100
#         if epoch % num == num - 1:
#             print('epochs:%d   MSEloss : %.3f' % (epoch + 1, run_loss / num))
#     return model_G


def gen_model_G_EX6(epochs=1000, num2in_point=100, num_inflowbd_points=50, region_l=0.0, region_r=1.0,
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
    for epoch in range(epochs):
        model_G.optimizer.zero_grad()
        inflow_boundary_points = rand_bd_inflow_EX6(num_inflowbd_points, region_a, region_b, init_l, init_r,
                                                    to_torch=False, to_float=True)
        x_train = inflow_boundary_points   # 用边界点作为输入
        y_train = g(inflow_boundary_points[:, 0], inflow_boundary_points[:,1]) # g(x,y)作为输出
        x_train, y_train = tensorv(x_train.reshape(-1, 2)), tensorv(y_train.reshape(-1,1))
        pred = model_G(x_train)
        loss = criterion(pred,y_train)
        loss.backward()
        model_G.optimizer.step()
        run_loss = loss.item()
        num = 100
        if epoch % num == num - 1:
            print('epochs:%d   MSEloss : %.6f' % (epoch + 1, run_loss / num))
    return model_G


if __name__ == "__main__":
    store_file = 'model_G'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR)
    seed = np.random.randint(1e5)
    seed_str = str(seed)  # int 型转为字符串型
    FolderName = BASE_DIR
    epoch = 10000
    batch_inside = 500
    batch_inflow = 5000
    region_a = 0
    region_b = 2
    init_l = 0
    init_r = 5
    # inside_points = rand_bd_ex1(batch_size=batch_inside, region_a=region_a, region_b=region_b, init_l=init_l, init_r=init_r, to_torch=False, to_float=True,
    #                to_cuda=False, gpu_no=0, use_grad=True)
    # inflow_points = rand_inflow(batch_size=batch_inflow, region_a=region_a, region_b=region_b, init_l=init_l,
    #                             init_r=init_r, to_torch=False, to_float
    #                             =True,
    #                             to_cuda=False, gpu_no=0, use_grad=True)
    # # label_norm = d(inside_points, inflow_points)
    # model_D = Gen_model_distance(epochs=epoch, num_inside_points=batch_inside, num_inflowbd_points=batch_inflow,
    #                            region_a=region_a, region_b=region_b, init_l=init_l, init_r=init_r)
    # torch.save(model_D.state_dict(), 'model_D111.pth')
    # plot_distance_function(model_D, region_a, region_b, init_l, init_r, splits=1000)


    ws = 0.001
    ds = 0.002
    pi = np.pi
    alpha = 0.1
    beta = 50
    zeta = 0.1
    # u_left = lambda x, t: np.zeros_like(x)
    # u_right = lambda x, t: np.zeros_likex
    # u_init = lambda x, t: np.sin(pi * x) + zeta * np.sin(beta * pi * x)
    # u_true = lambda x, t: np.multiply(np.exp(-alpha * t), model_G(x, t))
    model_G = lambda x, t: torch.sin(pi * x) + zeta * torch.sin(beta * pi * x)
    u_true = lambda x, t: torch.mul(torch.exp(-alpha * t), model_G(x, t))
    model_G = Gen_model_G(epochs=50000, num_inflowbd_points=batch_inflow * 10, region_a=region_a, region_b=region_b,
                          init_l=init_l, init_r=init_r, u_true=u_true)
    torch.save(model_G.state_dict(), 'model_G_111.pth')

    # plot
    # model_distance = Network()
    # state_dictD = torch.load('model_D100.pth',map_location='cpu')
    # model_distance.load_state_dict(state_dictD)
    # plot_distance_function(model_distance, region_a, region_b, init_l, init_r, splits=100)
    # splits =100
    # y, x = np.meshgrid(np.linspace(init_l, init_r, 100), np.linspace(region_a, region_b, 100))
    # points = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], 1)
    # points = tensorv(points)
    # points = points.cuda(device='cuda:' + str(0))
    # z = model_distance(points).detach().numpy()
    # a = z.reshape(splits, splits)
    # z_min, z_max = np.nanmin(z), np.nanmax(z)
    # fig, ax = plt.subplots(figsize=(13, 8))
    # ax.set_title('model_distance function')
    # c = ax.pcolormesh(x, y, a, cmap='RdBu', shading='auto', vmin=z_min, vmax=z_max)
    # # set the limits of the plot to the limits of the data
    # ax.axis([x.min(), x.max(), y.min(), y.max()])
    # fig.colorbar(c, ax=ax)
    # plt.savefig("distance" + seed_str + ".jpg")
    # plt.show()