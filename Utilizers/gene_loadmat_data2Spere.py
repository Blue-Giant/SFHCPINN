import numpy as np
import torch
import scipy
import scipy.stats.qmc as stqmc

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


# load the data from matlab of .mat
def load_Matlab_data(filename=None):
    data = scipy.io.loadmat(filename, mat_dtype=True, struct_as_record=True)  # variable_names='CATC'
    return data


def load_meshData2Spere_BD(self, path2file=None, num2point=2, to_float=True, shuffle_data=False):
    train_meshXY_file = path2file + str('TrainXYZbd1') + str('.mat')
    test_meshXY_file = path2file + str('TrainXYZbd2') + str('.mat')
    mesh2bd1 = load_Matlab_data(train_meshXY_file)
    mesh2bd2 = load_Matlab_data(test_meshXY_file)

    All_XYZ_bd1 = mesh2bd1['xyzbd1']
    All_XYZ_bd2 = mesh2bd2['xyzbd2']

    index = np.random.randint(low=0, high=22500, size=num2point)

    XYZ_bd1 = All_XYZ_bd1[index, :]
    XYZ_bd2 = All_XYZ_bd2[index, :]
    shape2XYZ = np.shape(XYZ_bd1)
    assert (len(shape2XYZ) == 2)
    if shape2XYZ[0] == 3:
        XYZ_bd1 = np.transpose(XYZ_bd1, (1, 0))
        XYZ_bd2 = np.transpose(XYZ_bd2, (1, 0))

    if to_float:
        XYZ_bd1 = XYZ_bd1.astype(dtype=self.float_type)
        XYZ_bd2 = XYZ_bd2.astype(dtype=self.float_type)
    if shuffle_data:
        np.random.shuffle(XYZ_bd1)
        np.random.shuffle(XYZ_bd2)
    return XYZ_bd1, XYZ_bd2


def gene_data2spere_inner_outer_bds(r_in=0.1, r_out=0.8, num2theta=100, num2gamma=20, opt2gene_data='mesh',
                                    opt2sampling='lhs', to_float=True, to_torch=True, to_cuda=False, gpu_no=0,
                                    use_grad2x=False):
    assert r_in < r_out
    if 'mesh' == opt2gene_data:
        theta2in = np.linspace(0.0, 2 * np.pi, num2theta)
        gamma2in = np.linspace(0.0, np.pi, num2gamma)

        theta2out = np.linspace(0.0, 2 * np.pi, num2theta)
        gamma2out = np.linspace(0.0, np.pi, num2gamma)
    elif 'lhs' == opt2sampling and 'mesh' != opt2gene_data:
        sampler = stqmc.LatinHypercube(d=1)
        theta2in = (2*np.pi - 0.0) * sampler.random(num2theta)
        gamma2in = (np.pi - 0.0) * sampler.random(num2gamma)

        theta2out = (2 * np.pi - 0.0) * sampler.random(num2theta)
        gamma2out = (np.pi - 0.0) * sampler.random(num2gamma)
    else:
        theta2in = (2 * np.pi - 0.0) * np.random.rand(num2theta, 1)
        gamma2in = (np.pi - 0.0) * np.random.rand(num2gamma, 1)

        theta2out = (2 * np.pi - 0.0) * np.random.rand(num2theta, 1)
        gamma2out = (np.pi - 0.0) * np.random.rand(num2gamma, 1)

    XX_in_bd = np.zeros(shape=[num2theta, num2gamma])
    YY_in_bd = np.zeros(shape=[num2theta, num2gamma])
    ZZ_in_bd = np.zeros(shape=[num2theta, num2gamma])

    XX_out_bd = np.zeros(shape=[num2theta, num2gamma])
    YY_out_bd = np.zeros(shape=[num2theta, num2gamma])
    ZZ_out_bd = np.zeros(shape=[num2theta, num2gamma])

    for i in range(num2theta):
        temp = r_in * np.sin(theta2in[i]) * np.cos(gamma2in)
        XX_in_bd[i, :] = np.reshape(r_in * np.sin(theta2in[i]) * np.cos(gamma2in), newshape=[-1])
        YY_in_bd[i, :] = np.reshape(r_in * np.sin(theta2in[i]) * np.sin(gamma2in), newshape=[-1])
        ZZ_in_bd[i, :] = np.reshape(r_in * np.cos(theta2in[i]), newshape=[-1])

        XX_out_bd[i, :] = np.reshape(r_out * np.sin(theta2out[i]) * np.cos(gamma2out), newshape=[-1])
        YY_out_bd[i, :] = np.reshape(r_out * np.sin(theta2out[i]) * np.sin(gamma2out), newshape=[-1])
        ZZ_out_bd[i, :] = np.reshape(r_out * np.cos(theta2out[i]), newshape=[-1])

    x_in_bd = np.reshape(XX_in_bd, newshape=[-1, 1])
    y_in_bd = np.reshape(YY_in_bd, newshape=[-1, 1])
    z_in_bd = np.reshape(ZZ_in_bd, newshape=[-1, 1])

    x_out_bd = np.reshape(XX_out_bd, newshape=[-1, 1])
    y_out_bd = np.reshape(YY_out_bd, newshape=[-1, 1])
    z_out_bd = np.reshape(ZZ_out_bd, newshape=[-1, 1])

    xyz_in_bd = np.concatenate([x_in_bd, y_in_bd, z_in_bd], axis=-1)
    xyz_out_bd = np.concatenate([x_out_bd, y_out_bd, z_out_bd], axis=-1)

    if to_float:
        xyz_in_bd = xyz_in_bd.astype(np.float32)
        xyz_out_bd = xyz_out_bd.astype(np.float32)

    if to_torch:
        xyz_in_bd = torch.from_numpy(xyz_in_bd)
        xyz_out_bd = torch.from_numpy(xyz_out_bd)

        if to_cuda:
            xyz_in_bd = xyz_in_bd.cuda(device='cuda:' + str(gpu_no))
            xyz_out_bd = xyz_out_bd.cuda(device='cuda:' + str(gpu_no))

        xyz_in_bd.requires_grad = use_grad2x
        xyz_out_bd.requires_grad = use_grad2x

    return xyz_in_bd, xyz_out_bd


def gene_data2spere_inner(r_in=0.1, r_out=0.8, num2radius=10, num2theta=100, num2gamma=20, opt2gene_data='mesh',
                          opt2sampling='lhs', to_float=True, to_torch=True, to_cuda=False, gpu_no=0,
                          use_grad2x=False):
    assert r_in < r_out
    if 'mesh' == opt2gene_data:
        radius = np.linspace(r_in, r_out, num2radius, endpoint=False)
        theta = np.linspace(0.0, 2 * np.pi, num2theta, endpoint=False)
        gamma = np.linspace(0.0, np.pi, num2gamma, endpoint=False)
    elif 'lhs' == opt2sampling and 'mesh' != opt2gene_data:
        sampler = stqmc.LatinHypercube(d=1)
        radius = (r_out - r_in) * sampler.random(num2radius) + r_in
        theta = (2*np.pi - 0.0) * sampler.random(num2theta)
        gamma = (np.pi - 0.0) * sampler.random(num2gamma)
    else:
        radius = (r_out - r_in) * np.random.rand(num2radius, 1) + r_in
        theta = (2 * np.pi - 0.0) * np.random.rand(num2theta, 1)
        gamma = (np.pi - 0.0) * np.random.rand(num2gamma, 1)

    XX_in = np.zeros(shape=[num2theta, num2gamma])
    YY_in = np.zeros(shape=[num2theta, num2gamma])
    ZZ_in = np.zeros(shape=[num2theta, num2gamma])

    for ir in range(num2radius):
        for i in range(num2theta):
            XX_in[ir, i, :] = np.reshape(radius[ir] * np.sin(theta[i]) * np.cos(gamma), newshape=[-1])
            YY_in[ir, i, :] = np.reshape(radius[ir] * np.sin(theta[i]) * np.sin(gamma), newshape=[-1])
            ZZ_in[ir, i, :] = np.reshape(radius[ir] * np.cos(theta[i]), newshape=[-1])

    x_in = np.reshape(XX_in, newshape=[-1, 1])
    y_in = np.reshape(YY_in, newshape=[-1, 1])
    z_in = np.reshape(ZZ_in, newshape=[-1, 1])

    xyz_in = np.concatenate([x_in, y_in, z_in], axis=-1)

    if to_float:
        xyz_in = xyz_in.astype(np.float32)

    if to_torch:
        xyz_in = torch.from_numpy(xyz_in)

        if to_cuda:
            xyz_in = xyz_in.cuda(device='cuda:' + str(gpu_no))

        xyz_in.requires_grad = use_grad2x

    return xyz_in


def gene_rand_data2spere_inner(r_in=0.1, r_out=0.8, num2points=10, opt2sampling='lhs', to_float=True, to_torch=True,
                               to_cuda=False, gpu_no=0, use_grad2x=False):
    assert r_in < r_out
    if 'lhs' == opt2sampling:
        sampler = stqmc.LatinHypercube(d=1)
        radius = (r_out - r_in) * sampler.random(num2points) + r_in
        theta = (2*np.pi - 0.0) * sampler.random(num2points)
        gamma = (np.pi - 0.0) * sampler.random(num2points)
    else:
        radius = (r_out - r_in) * np.random.rand(num2points, 1) + r_in
        theta = (2 * np.pi - 0.0) * np.random.rand(num2points, 1)
        gamma = (np.pi - 0.0) * np.random.rand(num2points, 1)

    x_in = np.reshape(radius * np.sin(theta) * np.cos(gamma), newshape=[-1, 1])
    y_in = np.reshape(radius * np.sin(theta) * np.sin(gamma), newshape=[-1, 1])
    z_in = np.reshape(radius * np.cos(theta), newshape=[-1, 1])

    # norm2point = np.sqrt(np.square(x_in) + np.square(y_in) + np.square(z_in))
    #
    # abs_x_in = np.abs(x_in)
    # abs_y_in = np.abs(y_in)
    # abs_z_in = np.abs(z_in)

    xyz_in = np.concatenate([x_in, y_in, z_in], axis=-1)

    if to_float:
        xyz_in = xyz_in.astype(np.float32)

    if to_torch:
        xyz_in = torch.from_numpy(xyz_in)

        if to_cuda:
            xyz_in = xyz_in.cuda(device='cuda:' + str(gpu_no))

        xyz_in.requires_grad = use_grad2x

    return xyz_in


def gene_data2inner_spere_slice(r_slice=0.5, num2theta=100, num2gamma=20, opt2gene_data='mesh',
                                opt2sampling='lhs', to_float=True, to_torch=True, to_cuda=False, gpu_no=0,
                                use_grad2x=False):
    if 'mesh' == opt2gene_data:
        theta = np.linspace(0.0, 2 * np.pi, num2theta, endpoint=False)
        gamma = np.linspace(0.0, np.pi, num2gamma, endpoint=False)
    elif 'lhs' == opt2sampling and 'mesh' != opt2gene_data:
        sampler = stqmc.LatinHypercube(d=1)
        theta = (2*np.pi - 0.0) * sampler.random(num2theta)
        gamma = (np.pi - 0.0) * sampler.random(num2gamma)
    else:
        theta = (2 * np.pi - 0.0) * np.random.rand(num2theta, 1)
        gamma = (np.pi - 0.0) * np.random.rand(num2gamma, 1)

    XX_slice = np.zeros(shape=[num2theta, num2gamma])
    YY_slice = np.zeros(shape=[num2theta, num2gamma])
    ZZ_slice = np.zeros(shape=[num2theta, num2gamma])

    for i in range(num2theta):
        XX_slice[i, :] = np.reshape(r_slice * np.sin(theta[i]) * np.cos(gamma), newshape=[-1])
        YY_slice[i, :] = np.reshape(r_slice * np.sin(theta[i]) * np.sin(gamma), newshape=[-1])
        ZZ_slice[i, :] = np.reshape(r_slice * np.cos(theta[i]), newshape=[-1])

    x_slice = np.reshape(XX_slice, newshape=[-1, 1])
    y_slice = np.reshape(YY_slice, newshape=[-1, 1])
    z_slice = np.reshape(ZZ_slice, newshape=[-1, 1])

    xyz_slice = np.concatenate([x_slice, y_slice, z_slice], axis=-1)

    if to_float:
        xyz_slice = xyz_slice.astype(np.float32)

    if to_torch:
        xyz_slice = torch.from_numpy(xyz_slice)

        if to_cuda:
            xyz_slice = xyz_slice.cuda(device='cuda:' + str(gpu_no))

        xyz_slice.requires_grad = use_grad2x

    return xyz_slice


def load_meshData2Spere(path2file=None, mesh_number=2, to_float=True, to_torch=False, to_cuda=False, gpu_no=0,
                        shuffle_data=False, use_grad2x=False):
    if shuffle_data:
        test_meshXY_file = path2file + str('Shufle_TestXYZ') + str('.mat')
        mesh_points = load_Matlab_data(test_meshXY_file)
        mesh_XYZ = mesh_points['XYZ_shufle']
    else:
        test_meshXY_file = path2file + str('TestXYZ') + str('.mat')
        mesh_points = load_Matlab_data(test_meshXY_file)
        mesh_XYZ = mesh_points['XYZ']

    shape2XYZ = np.shape(mesh_XYZ)
    assert (len(shape2XYZ) == 2)
    if shape2XYZ[0] == 3:
        xyz_mesh = np.transpose(mesh_XYZ, (1, 0))
    else:
        xyz_mesh = mesh_XYZ

    if to_float:
        xyz_mesh = xyz_mesh.astype(dtype=np.float32)

    if to_torch:
        xyz_mesh = torch.from_numpy(xyz_mesh)

        if to_cuda:
            xyz_mesh = xyz_mesh.cuda(device='cuda:' + str(gpu_no))

        xyz_mesh.requires_grad = use_grad2x


    return xyz_mesh


def test_spere_bd():
    xyz_inner_bd, xyz_outer_bd = gene_data2spere_inner_outer_bds(
        r_in=0.1, r_out=0.8, num2theta=100, num2gamma=80, opt2gene_data='random', opt2sampling='random', to_float=True,
        to_torch=False, to_cuda=False, gpu_no=0, use_grad2x=False)
    # 绘制解的3D散点图(真解和预测解)
    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    ax.scatter(xyz_inner_bd[:, 0], xyz_inner_bd[:, 1], xyz_inner_bd[:, 2], c='b', label='inner')
    # ax.scatter(test_x_bach, test_y_bach, solu2_test, c='b', label=actName2)

    # 绘制图例
    ax.legend(loc='best')
    # 添加坐标轴(顺序是X，Y, Z)
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('u', fontdict={'size': 15, 'color': 'red'})
    plt.show()


def test2gene_rand_points():
    xyz = gene_rand_data2spere_inner(r_in=0.1, r_out=0.8, num2points=5000, opt2sampling='lhs', to_float=True, to_torch=False,
                                     to_cuda=False, gpu_no=0, use_grad2x=False)


if __name__ == "__main__":
    # test_spere_bd()
    test2gene_rand_points()
