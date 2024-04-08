import torch
from Networks.NETWORK import *
from Utilizers.gen_data import *
import matplotlib.pyplot as plt


def d(xs, xbs):
    # xs: List of collocation points
    # xbs: List of boundary conditions
    xs, xbs = np.array(xs), np.array(xbs)
    ds = [min([np.linalg.norm(x - xb) for xb in xbs]) for x in xs]
    return ds


def gen_xy_distance(num_inside_points, num_inflowbd_points, region_a, region_b, init_l, init_r, variable_dim=2, ):
    random_inside_points = rand_bd_inside(num_inside_points, variable_dim, region_a, region_b, init_l, init_r,
                                          to_torch=False, to_float=True)

    random_boundary_points = rand_bd_inflow(num_inflowbd_points, variable_dim, region_a, region_b, init_l, init_r,
                                            to_torch=False, to_float=True)
    # Random set to train of train_size random points in the domain (excluding extremal points)
    x_train = np.array(random_inside_points).reshape(-1, 2)  # random_inside points是内部随机点
    ds = d(x_train, random_boundary_points)  # 内部点和inflow边界点的距离
    y_train = np.array(ds).reshape(-1, 1)  # y_train 是内部点和inflow边界点的距离 要使这个最小
    x_train, y_train = tensorv(x_train), tensorv(y_train)
    return x_train, y_train


def Gen_model_distance(epochs, num_inside_points, num_inflowbd_points, region_a, region_b, init_l, init_r):
    model_distance = Network()
    print('---------start training Model D')
    for epoch in range(epochs):
        model_distance.optimizer.zero_grad()
        x_train, y_train = gen_xy_distance(num_inside_points, num_inflowbd_points, region_a, region_b, init_l, init_r)
        pred = model_distance(x_train)
        loss = model_distance.criterion(pred, y_train)
        loss.backward()
        model_distance.optimizer.step()
        run_loss = loss.item()
        num = 100
        if epoch % num == num - 1:
            print('epochs:%d   MSEloss : %.3f' % (epoch + 1, run_loss / num))
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


def Gen_model_G(epochs, num_inflowbd_points, region_a, region_b, init_l, init_r, g):
    model_G = Network()
    print('---------start training Model G')
    for epoch in range(epochs):
        model_G.optimizer.zero_grad()
        inflow_boundary_points = rand_bd_inflow(num_inflowbd_points, 2, region_a, region_b, init_l, init_r,
                                                to_torch=True, to_float=True)
        x_train = inflow_boundary_points  # 用边界点作为输入
        y_train = g(inflow_boundary_points[:, 0], inflow_boundary_points[:, 1])  # g(x,y)作为输出
        x_train, y_train = tensorv(x_train), tensorv(y_train)
        pred = model_G(x_train)
        loss = model_G.criterion(pred, y_train)
        loss.backward()
        model_G.optimizer.step()
        run_loss = loss.item()
        num = 100
        if epoch % num == num - 1:
            print('epochs:%d   MSEloss : %.3f' % (epoch + 1, run_loss / num))
    return model_G


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


def gen_xy_distance1(num_inside_points, num_inflowbd_points, region_a, region_b, init_l, init_r, variable_dim=2):
    random_inside_points = rand_bd_inside(num_inside_points, variable_dim, region_a, region_b, init_l, init_r,
                                          to_torch=False, to_float=True)
    x_train = np.array(random_inside_points).reshape(-1, 2)  # random_inside points是内部随机点
    random_inside_points = scaling(random_inside_points, region_a, region_b, init_l, init_r).reshape(-1, 2)

    random_boundary_points = rand_bd_inflow(num_inflowbd_points, variable_dim, region_a, region_b, init_l, init_r,
                                            to_torch=False, to_float=True)
    random_boundary_points = scaling(random_boundary_points, region_a, region_b, init_l, init_r)
    # Random set to train of train_size random points in the domain (excluding extremal points)

    ds = d(random_inside_points, random_boundary_points)  # 内部点和inflow边界点的距离
    y_train = np.array(ds).reshape(-1, 1)  # y_train 是内部点和inflow边界点的距离 要使这个最小
    x_train, y_train = tensorv(x_train), tensorv(y_train)
    return x_train, y_train


def scaling(x_in_batch, region_l, region_r, init_time, end_time):
    x_in_batch[:, 0] = (x_in_batch[:, 0] - region_l) / (region_r - region_l)
    x_in_batch[:, 1] = (x_in_batch[:, 1] - init_time) / (end_time - init_time)
    return x_in_batch


def Gen_model_distance2(epochs, num_inside_points, num_inflowbd_points, region_a, region_b, init_l, init_r):
    model_distance = Network()
    print('---------start training Model D')
    for epoch in range(epochs):
        model_distance.optimizer.zero_grad()
        x_train, y_train = gen_xy_distance2(num_inside_points, num_inflowbd_points, region_a, region_b, init_l, init_r)
        pred = model_distance(x_train)
        loss = model_distance.criterion(pred, y_train)
        loss.backward()
        model_distance.optimizer.step()
        run_loss = loss.item()
        num = 100
        if epoch % num == num - 1:
            print('epochs:%d   MSEloss : %.6f' % (epoch + 1, run_loss / num))
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
    ds = d(random_points, random_boundary_points)  # 内部点和inflow边界点的距离
    y_train = np.array(ds).reshape(-1, 1)  # y_train 是内部点和inflow边界点的距离 要使这个最小
    x_train, y_train = tensorv(x_train), tensorv(y_train)
    return x_train, y_train


def Gen_model_distance_EX6(epochs, num_inside_points, num_inflowbd_points, region_a, region_b, init_l, init_r,
                           SCALE=False):
    model_distance = Network()
    print('---------start training Model D')
    for epoch in range(epochs):
        model_distance.optimizer.zero_grad()
        x_train, y_train = gen_xy_distance_EX6(num_inside_points, num_inflowbd_points, region_a, region_b, init_l,
                                               init_r, norm=SCALE)
        pred = model_distance(x_train)
        loss = model_distance.criterion(pred, y_train)
        loss.backward()
        model_distance.optimizer.step()
        run_loss = loss.item()
        num = 100
        if epoch % num == num - 1:
            print('epochs:%d   MSEloss : %.6f' % (epoch + 1, run_loss / num))
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
        ds = d(inputs, random_boundary_points)  # 内部点和inflow边界点的距离

    else:
        random_inside_points = rand_bd_inside_EX6(num_inside_points, region_a, region_b, init_l, init_r,
                                                  to_torch=False, to_float=True)
        random_boundary_points = rand_bd_inflow_EX6(num_inflowbd_points, region_a, region_b, init_l, init_r,
                                                    to_torch=False, to_float=True)
        # Random set to train of train_size random points in the domain (excluding extremal points)
        x_train = np.concatenate([random_boundary_points, random_inside_points], 0).reshape(-1, 2)
        # random_inside points是内部随机点 加randon_boundary_points是为了满足&加深inflow边界点为0这个边界

        ds = d(x_train, random_boundary_points)  # 内部点和inflow边界点的距离

    y_train = np.array(ds).reshape(-1, 1)  # y_train 是内部点和inflow边界点的距离 要使这个最小
    x_train, y_train = tensorv(x_train), tensorv(y_train)
    return x_train, y_train


def gen_model_G_EX6(num_epoches,num_inflowbd_points, region_a, region_b,init_l, init_r,g,criterion = torch.nn.MSELoss()):
    model_G = Network()
    for epoch in range(num_epoches):
        model_G.optimizer.zero_grad()
        inflow_boundary_points = rand_bd_inflow_EX6(num_inflowbd_points, region_a, region_b,init_l, init_r,
                                                           to_torch=False,to_float=True)
        x_train = inflow_boundary_points #用边界点作为输入
        y_train = g(inflow_boundary_points[:,0],inflow_boundary_points[:,1]) # g(x,y)作为输出
        x_train ,y_train = tensorv(x_train.reshape(-1,2)),tensorv(y_train.reshape(-1,1))
        pred = model_G(x_train)
        loss = criterion(pred,y_train)
        loss.backward()
        model_G.optimizer.step()
        run_loss = loss.item()
        num = 100
        if epoch % num == num - 1:
            print('epochs:%d   MSEloss : %.6f' % ( epoch + 1, run_loss / num))
    return model_G