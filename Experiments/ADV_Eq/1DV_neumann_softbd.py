"""
@author: LXA
 Date: 2021 年 11 月 11 日
 Modifying on 2022 年 9月 2 日 ~~~~ 2022 年 9月 12 日
@cooperate author:DJX
 Date :2022 年 10月
"""
import os
import sys
import torch
import torch.nn as tn
import numpy as np
import matplotlib
import platform
import shutil
import time
import datetime

from Networks import DNN_base
from Utilizers import DNN_tools
from Utilizers import dataUtilizer2torch

from Utilizers import saveData
from Utilizers import plotData
from Utilizers import DNN_Log_Print
from Utilizers.Load_data2Mat import *
import torchvision
from Utilizers.gen_points import *


def get_fside2general(x,  t, alpha=0.25, beta=1.0, ds=0.01, ws=0.01, PI=torch.pi):
    # utrue = torch.mul(torch.exp(-alpha * t), torch.sin(PI*x))
    exp_t = torch.exp(-alpha * t)

    pu_pt = -alpha*exp_t*torch.sin(PI*x)

    temp2pu_px = PI*torch.cos(PI*x)
    pu_px = exp_t*temp2pu_px

    temp2ppu_pxx = -PI*PI*torch.sin(PI*x)
    ppu_pxx = exp_t * temp2ppu_pxx

    f_side = pu_pt - ds * ppu_pxx + ws * pu_px
    return f_side


def grad_fun(model_D, model_G, XY):
    # 拓展函数的梯度
    X = XY[:, 0].reshape(-1, 1)
    GNN = model_G(XY).reshape(-1, 1)
    grad2GNN = torch.autograd.grad(GNN, XY, grad_outputs=torch.ones_like(X), create_graph=True,
                                   retain_graph=True)
    dGNN = grad2GNN[0]
    dGNN2x = torch.reshape(dGNN[:, 0], shape=[-1, 1])
    dGNN2y = torch.reshape(dGNN[:, 1], shape=[-1, 1])
    dGNNxxy = torch.autograd.grad(dGNN2x, XY, grad_outputs=torch.ones_like(X), create_graph=True,
                                  retain_graph=True)[0]
    dGNNxx = torch.reshape(dGNNxxy[:, 0], shape=[-1, 1])

    # 初始化距离函数的一些值
    DNN = model_D(XY).reshape(-1, 1)
    grad2DNN = torch.autograd.grad(DNN, XY, grad_outputs=torch.ones_like(X), create_graph=True,
                                   retain_graph=True)
    dDNN = grad2DNN[0]
    dDNN2x = torch.reshape(dDNN[:, 0], shape=[-1, 1])
    dDNN2y = torch.reshape(dDNN[:, 1], shape=[-1, 1])
    dDNNxxy = torch.autograd.grad(dDNN2x, XY, grad_outputs=torch.ones_like(X), create_graph=True,
                                  retain_graph=True)[0]
    dDNNxx = torch.reshape(dDNNxxy[:, 0], shape=[-1, 1])
    return DNN, dDNN2x, dDNNxx, dDNN2y, GNN, dGNN2x, dGNNxx, dGNN2y


def grad_fun_nmbd(DNN, XY_bd):
    X = torch.reshape(XY_bd[:, 0], shape=[-1, 1])
    UNN = DNN(XY_bd).reshape(-1, 1)
    grad2UNN = torch.autograd.grad(UNN, XY_bd, grad_outputs=torch.ones_like(X), create_graph=True,
                                   retain_graph=True)
    dUNN = grad2UNN[0]
    dUNN2x = torch.reshape(dUNN[:, 0], shape=[-1, 1])
    return UNN, dUNN2x


def MinMaxScalling(a):
    # Min-Max scaling
    min_a = torch.min(a)
    max_a = torch.max(a)
    n2 = (a - min_a) / (max_a - min_a)
    return n2


def get_fside2multi_scale(x, t, alpha=0.25, zeta=0.1, omega=10, ws=0.1, ds=0.1, PI=torch.pi):
    exp_t = torch.exp(-alpha * t)
    sin_PIx_sinOmegax = torch.sin(PI * x) + zeta * torch.sin(omega * PI * x)
    ut = -alpha*torch.mul(exp_t, sin_PIx_sinOmegax)
    ux = torch.mul(exp_t, PI*torch.cos(PI*x) + zeta*omega*PI*torch.cos(omega*PI*x))
    uxx = torch.mul(exp_t, -1.0*PI*PI*torch.sin(PI*x)-1.0*zeta*omega*omega*PI*PI*torch.sin(omega*PI*x))
    f = ut - ds*uxx + ws*ux
    return f


class SoftPINN(tn.Module):
    def __init__(self, input_dim=4, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='relu',
                 name2actHidden='relu', name2actOut='linear', opt2regular_WB='L2', type2numeric='float32',
                 factor2freq=None, sFourier=1.0, use_gpu=False, No2GPU=0, repeat_highFreq=True):
        super(SoftPINN, self).__init__()
        if 'DNN' == str.upper(Model_name):
            self.DNN = DNN_base.Pure_FullyNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'SCALE_DNN' == str.upper(Model_name) or 'DNN_SCALE' == str.upper(Model_name):
            self.DNN = DNN_base.Fully_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'FOURIER_DNN' == str.upper(Model_name) or 'DNN_FOURIERBASE' == str.upper(Model_name):
            self.DNN = DNN_base.Fully_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'FOURIER_SUBDNN' == str.upper(Model_name) or 'SUBDNN_FOURIERBASE' == str.upper(Model_name):
            self.DNN = DNN_base.Fourier_SubNets3D(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                num2subnets=len(factor2freq), to_gpu=use_gpu, gpu_no=No2GPU)

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_layer = hidden_layer
        self.Model_name = Model_name
        self.name2actIn = name2actIn
        self.name2actHidden = name2actHidden
        self.name2actOut = name2actOut
        self.factor2freq = factor2freq
        self.sFourier = sFourier
        self.opt2regular_WB = opt2regular_WB

        if type2numeric == 'float32':
            self.float_type = torch.float32
        elif type2numeric == 'float64':
            self.float_type = torch.float64
        elif type2numeric == 'float16':
            self.float_type = torch.float16

        self.use_gpu = use_gpu
        if use_gpu:
            self.opt2device = 'cuda:' + str(No2GPU)
        else:
            self.opt2device = 'cpu'
        self.mat2X = torch.tensor([[1, 0]], dtype=self.float_type, device=self.opt2device)  # 1 行 2 列
        self.mat2U = torch.tensor([[0, 1]], dtype=self.float_type, device=self.opt2device)  # 1 行 2 列
        self.mat2T = torch.tensor([[0, 1]], dtype=self.float_type, device=self.opt2device)  # 1 行 3 列

    def loss_1DV(self, X_in=None, T_in=None, loss_type='l2_loss', scale2lncosh=0.5,
                 ws=None, ds=None, force_side=None, if_lambda2fside=True):
        '''
        Args:
            X_in: 输入的内部深度（x or z)
            T_in: 输入的内部时间 (t)
            loss_type:
            scale2lncosh:
            ws: dx 的常系数
            ds: dxx 的常系数
        Returns:

        '''
        # 判断输入的内部点不为None
        assert (X_in is not None)
        assert (T_in is not None)

        # 判断 X_in 的形状
        shape2X = X_in.shape
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        # 判断 T_in 的形状
        shape2T = T_in.shape
        lenght2T_shape = len(shape2T)
        assert (lenght2T_shape == 2)
        assert (shape2T[-1] == 1)

        XT = torch.mul(X_in, self.mat2X) +torch.mul(T_in, self.mat2T)
        if if_lambda2fside:
            Fside = force_side(X_in, T_in)
        else:
            Fside = force_side
        # Fside = f_side(X_in, T_in)

        # 计算内部点损失 dx dxx
        UNN = self.DNN(XT, scale=self.factor2freq, sFourier=self.sFourier)
        grad2UNN = torch.autograd.grad(UNN, X_in, grad_outputs=torch.ones_like(X_in),
                                       create_graph=True, retain_graph=True)
        dUNN2x = grad2UNN[0]
        dUNNxx = torch.autograd.grad(dUNN2x, X_in, grad_outputs=torch.ones_like(X_in),
                                     create_graph=True, retain_graph=True)[0]
        # dt
        dUNN2t = torch.autograd.grad(UNN, T_in, grad_outputs=torch.ones_like(T_in),
                                     create_graph=True, retain_graph=True)[0]
        res = dUNN2t + ws * dUNN2x - ds * dUNNxx - Fside
        if str.lower(loss_type) == 'l2_loss':
            loss_it = torch.mean(torch.square(res))
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_it = (1 / scale2lncosh) * torch.mean(torch.log(torch.cosh(scale2lncosh * res)))
        return UNN, loss_it

    def loss2bd(self, X_bd=None, T_bd=None, Ubd_exact=None, if_lambda2Ubd=True, loss_type='l2_loss', scale2lncosh=0.2):
        # 判断输入的内部点不为None
        assert (X_bd is not None)
        assert (T_bd is not None)

        # 判断 X_in 的形状
        shape2X = X_bd.shape
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        # 判断 T_in 的形状
        shape2T = T_bd.shape
        lenght2T_shape = len(shape2T)
        assert (lenght2T_shape == 2)
        assert (shape2T[-1] == 1)

        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd, T_bd)
        else:
            Ubd = Ubd_exact

        XT_bd = torch.mul(X_bd, self.mat2X) + torch.mul(T_bd, self.mat2T)

        UNN_bd = self.DNN(XT_bd, scale=self.factor2freq, sFourier=self.sFourier)
        diff_bd = UNN_bd - Ubd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd = torch.mean(torch.square(diff_bd))
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_bd = (1 / scale2lncosh) * torch.mean(torch.log(torch.cosh(scale2lncosh * diff_bd)))
        return loss_bd

    def loss2Neu(self, X_bd=None, T_bd=None, Ubd_exact=None, if_lambda2Ubd=True, loss_type='l2_loss'):
        # 判断输入的内部点不为None
        assert (X_bd is not None)
        assert (T_bd is not None)

        # 判断 X_in 的形状
        shape2X = X_bd.shape
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        # 判断 T_in 的形状
        shape2T = T_bd.shape
        lenght2T_shape = len(shape2T)
        assert (lenght2T_shape == 2)
        assert (shape2T[-1] == 1)

        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd, T_bd)
        else:
            Ubd = Ubd_exact

        XT_bd = torch.mul(X_bd, self.mat2X) + torch.mul(T_bd, self.mat2T)

        # 计算内部点损失 dx dxx
        UNN = self.DNN(XT_bd, scale=self.factor2freq, sFourier=self.sFourier)
        grad2UNN = torch.autograd.grad(UNN, X_bd, grad_outputs=torch.ones_like(X_bd),
                                       create_graph=True, retain_graph=True)
        dUNN2x = grad2UNN[0]

        res = dUNN2x - Ubd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd = torch.mean(torch.square(res))
        return loss_bd

    def get_regularSum2WB(self):
        sum2WB = self.DNN.get_regular_sum2WB(self.opt2regular_WB)
        return sum2WB

    def eval_SoftPINN(self, XY_points=None):
        assert (XY_points is not None)
        shape2XY = XY_points.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        UNN = self.DNN(XY_points, scale=self.factor2freq, sFourier=self.sFourier)
        return UNN


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']  # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)  # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2train', R['name2act_hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Log_Print.dictionary_out2file(R, log_fileout)

    # 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']
    batchsize_init =R['batch_size2init']

    bd_penalty_init = R['init_boundary_penalty']  # Regularization parameter for boundary conditions
    init_penalty_init = R['init_penalty2init']

    penalty2WB = R['penalty2weight_biases']  # Regularization parameter for weights and biases
    learning_rate = R['learning_rate']

    input_dim = R['input_dim']
    out_dim = R['output_dim']
    R['PDE_type'] = '1DV'
    if R['equa_name'] == '1DV_General':
        region_l = 0.0
        region_r = 2.0
        init_time = 0
        end_time = 5
        beta = 1
        ws = 1.0
        ds = 0.1
        alpha = 0.25  # 二次项
        u_true = lambda x, t: torch.mul(torch.exp(-alpha * t), torch.sin(torch.pi * x))
        u_left = lambda x, t: torch.exp(-alpha * t) * torch.pi * torch.cos(torch.pi * x)
        u_right = lambda x, t: torch.exp(-alpha * t) * torch.pi * torch.cos(torch.pi * x)
        u_init = lambda x, t: torch.sin(torch.pi * x)
        f_side = lambda x, t: get_fside2general(x, t, alpha=alpha, beta=1.0, ds=ds, ws=ws)
    elif R['equa_name'] == '1DV_MultiScale':
        region_l = 0.0
        region_r = 1.0
        init_time = 0.0
        end_time = 1.0
        beta = 1
        alpha = 0.25  # 二次项
        omega = 10
        zeta = 0.2
        ws = 1.0
        ds = 0.1
        u_true = lambda x, t: torch.mul(torch.exp(-alpha*t), torch.sin(torch.pi*x)+zeta*torch.sin(omega*torch.pi*x))
        u_left = lambda x, t: torch.exp(-alpha * t) * (torch.pi*torch.cos(torch.pi*x)+zeta*omega*torch.pi*torch.cos(omega*torch.pi*x))
        u_right = lambda x, t: torch.exp(-alpha * t) * (torch.pi*torch.cos(torch.pi*x)+zeta*omega*torch.pi*torch.cos(omega*torch.pi*x))
        u_init = lambda x, t: torch.sin(torch.pi*x)+zeta*torch.sin(omega*torch.pi*x)
        f_side = lambda x, t: get_fside2multi_scale(x, t, alpha=alpha, zeta=zeta, omega=omega, ws=ws, ds=ds)

    model = SoftPINN(input_dim=R['input_dim'] + 1, out_dim=R['output_dim'], hidden_layer=R['hidden_layers'],
                     Model_name=R['model2NN'], name2actIn=R['name2act_in'], name2actHidden=R['name2act_hidden'],
                     name2actOut=R['name2act_out'], opt2regular_WB='L0', repeat_highFreq=R['repeat_highFreq'],
                     type2numeric='float32', factor2freq=R['freq'], use_gpu=R['use_gpu'], No2GPU=R['gpuNo'])
    if True == R['use_gpu']:
        model = model.cuda(device='cuda:' + str(R['gpuNo']))

    params2Net = model.DNN.parameters()

    # 定义优化方法，并给定初始学习率
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr)                     # SGD
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr, momentum=0.8)       # momentum
    # optimizer = torch.optim.RMSprop(params2Net, lr=init_lr, alpha=0.95)     # RMSProp
    optimizer = torch.optim.Adam(params2Net, lr=learning_rate)  # Adam

    # 定义更新学习率的方法
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.995)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.975)

    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    test_epoch = []

    if R['testData_model'] == 'random_generate':
        # 生成测试数据，用于测试训练后的网络
        test_bach_size = 4900

        Xpoints = (region_r - region_l) * np.random.random(size=[test_bach_size, 1]) + region_l
        Tpoints = (end_time - init_time) * np.random.random(size=[test_bach_size, 1]) + init_time
        test_xy_bach = np.concatenate((Xpoints, Tpoints), axis=-1)
        test_xy_bach = test_xy_bach.astype(np.float32)
    elif R['testData_model'] == 'generate_mesh':
        test_size2mesh = 70
        test_x_batch = np.linspace(region_l, region_r, test_size2mesh, dtype=np.float32).reshape(-1, 1)
        test_t_batch = np.linspace(init_time, end_time, test_size2mesh, dtype=np.float32).reshape(-1, 1)
        meshX, meshT = np.meshgrid(test_x_batch, test_t_batch)  # Construct the mesh points

        test_xy_bach = np.concatenate((np.reshape(meshX, newshape=[-1, 1]),
                                       np.reshape(meshT, newshape=[-1, 1])), axis=-1)
        np.random.shuffle(test_xy_bach)
    elif R['testData_model'] == 'load_MatData':
        if R['equa_name'] == '1DV_General':
            mat_data_path = '../OneDimMatData2ADE/ST_02_05/'
        elif R['equa_name'] == '1DV_MultiScale':
            mat_data_path = '../OneDimMatData2ADE/ST_01_01/'
        test_xy_bach = getMeshMatData2Space_Time(base_path=mat_data_path, dim=2, mesh_size=7, to_float=True,
                                                 to_torch=False, to_cuda=False, gpu_no=0, use_grad=False)

    saveData.save_testData_or_solus2mat(test_xy_bach, dataName='testXY', outPath=R['FolderName'])

    test_xy_bach = test_xy_bach.astype(np.float32)
    test_xy_torch = torch.from_numpy(test_xy_bach).reshape(-1, 2)
    # test_xy_bach.requires_grad_(True)
    if True == R['use_gpu']:
        test_xy_torch = test_xy_torch.cuda(device='cuda:' + str(R['gpuNo']))

    t0 = time.time()
    for i_epoch in range(R['max_epoch'] + 1):
        if R['PDE_type'] == '1DV':
            x_in_batch = dataUtilizer2torch.rand_in_1D(
                batch_size=batchsize_it, variable_dim=input_dim, region_a=region_l, region_b=region_r, to_torch=True,
                to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'], use_grad2x=True)
            t_in_batch = dataUtilizer2torch.rand_in_1D(
                batch_size=batchsize_it, variable_dim=1, region_a=init_time, region_b=end_time, to_torch=True,
                to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'], use_grad2x=True)
            t_bd_batch = dataUtilizer2torch.rand_in_1D(
                batch_size=batchsize_bd, variable_dim=1, region_a=init_time, region_b=end_time, to_torch=True,
                to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'], use_grad2x=True)
            x_init_batch = dataUtilizer2torch.rand_in_1D(
                batch_size=batchsize_init, variable_dim=input_dim, region_a=region_l, region_b=region_r, to_torch=True,
                to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'], use_grad2x=True)
            t_init_batch = torch.ones_like(x_init_batch) * 0
            x_lbd_batch = torch.ones_like(t_bd_batch) * region_l
            x_rbd_batch = torch.ones_like(t_bd_batch) * region_r
            x_lbd_batch.requires_grad_(True)
            x_rbd_batch.requires_grad_(True)
            t_init_batch.requires_grad_(True)

            if R['activate_penalty2bd_increase'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 10 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 50 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 100 * bd_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 200 * bd_penalty_init
                else:
                    temp_penalty_bd = 500 * bd_penalty_init
            else:
                temp_penalty_bd = bd_penalty_init

            if R['activate_penalty2init_increase'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_init = init_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_init = 10 * init_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_init = 50 * init_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_init = 100 * init_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_init = 200 * init_penalty_init
                else:
                    temp_penalty_init = 500 * init_penalty_init
            else:
                temp_penalty_init = init_penalty_init
            UNN2train, loss_it = model.loss_1DV(X_in=x_in_batch, T_in=t_in_batch, loss_type='l2_loss', scale2lncosh=0.5,
                                                ws=beta, ds=alpha, force_side=f_side)

            loss_bd2left = model.loss2Neu(X_bd=x_lbd_batch, T_bd=t_bd_batch, Ubd_exact=u_left, if_lambda2Ubd=True, loss_type='l2_loss')
            loss_bd2right = model.loss2Neu(X_bd=x_rbd_batch, T_bd=t_bd_batch, Ubd_exact=u_right, if_lambda2Ubd=True, loss_type='l2_loss')
            loss_init = model.loss2bd(X_bd=x_init_batch, T_bd=t_init_batch, Ubd_exact=u_init, if_lambda2Ubd=True, loss_type='l2_loss')

            loss_bd = loss_bd2left + loss_bd2right
            # PWB = penalty2WB * mscalednn.get_regularSum2WB()
            loss = loss_it + loss_bd * temp_penalty_bd + loss_init * temp_penalty_init
            Uexact2train = u_true(torch.reshape(x_in_batch, shape=[-1, 1]),
                                  torch.reshape(t_in_batch, shape=[-1, 1]))
            train_mse = torch.mean(torch.square(UNN2train - Uexact2train))
            train_rel = train_mse / torch.mean(torch.square(Uexact2train))

        loss_it_all.append(loss_it.item())
        loss_bd_all.append(loss_bd.item())
        loss_all.append(loss.item())

        optimizer.zero_grad()  # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()  # 对loss关于Ws和Bs求偏导
        optimizer.step()  # 更新参数Ws和Bs
        scheduler.step()

        if R['PDE_type'] == '1DV':
            Uexact2train = u_true(torch.reshape(x_in_batch, shape=[-1, 1]),
                                  torch.reshape(t_in_batch, shape=[-1, 1])).reshape(-1, 1)
            train_mse = torch.mean(torch.square(UNN2train - Uexact2train))
            train_rel = train_mse / torch.mean(torch.square(Uexact2train))

        train_mse_all.append(train_mse.item())
        train_rel_all.append(train_rel.item())

        if i_epoch % 1000 == 0:
            run_times = time.time() - t0
            tmp_lr = optimizer.param_groups[0]['lr']
            DNN_tools.print_and_log_train_one_epoch_pinn2hard(i_epoch, run_times, tmp_lr, loss_it.item(), loss_bd.item(),
                                                           loss.item(), train_mse.item(), train_rel.item(),
                                                           log_out=log_fileout)

            test_epoch.append(i_epoch / 1000)
            if R['PDE_type'] == '1DV':
                UNN2test = model.eval_SoftPINN(XY_points=test_xy_torch).reshape(-1, 1)
                Utrue2test = u_true(torch.reshape(test_xy_torch[:, 0], shape=[-1, 1]),
                                    torch.reshape(test_xy_torch[:, 1], shape=[-1, 1]))
            else:
                UNN2test = model.eval_SoftPINN(XY_points=test_xy_torch) * 0
                Utrue2test = u_true(torch.reshape(test_xy_torch[:, 0], shape=[-1, 1]),
                                    torch.reshape(test_xy_torch[:, 1], shape=[-1, 1]))

            point_square_error = torch.square(Utrue2test - UNN2test)
            test_mse = torch.mean(point_square_error)
            test_rel = test_mse / torch.mean(torch.square(Utrue2test))
            test_mse_all.append(test_mse.item())
            test_rel_all.append(test_rel.item())
            DNN_tools.print_and_log_test_one_epoch(test_mse.item(), test_rel.item(), log_out=log_fileout)

    # ------------------- save the training results into mat file and plot them -------------------------
    saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=R['name2act_hidden'],
                                         outPath=R['FolderName'])
    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=R['name2act_hidden'], outPath=R['FolderName'])

    plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', outPath=R['FolderName'])

    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=R['name2act_hidden'], outPath=R['FolderName'])
    plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=R['name2act_hidden'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    # ----------------------  save testing results to mat files, then plot them --------------------------------
    saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=R['name2act_hidden'], outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=R['name2act_hidden'],
                              outPath=R['FolderName'], yaxis_scale=True)
    if True == R['use_gpu']:
        utrue2test_numpy = Utrue2test.cpu().detach().numpy()
        unn2test_numpy = UNN2test.cpu().detach().numpy()
        point_square_error_numpy = point_square_error.cpu().detach().numpy()
    else:
        utrue2test_numpy = Utrue2test.detach().numpy()
        unn2test_numpy = UNN2test.detach().numpy()
        point_square_error_numpy = point_square_error.detach().numpy()

    saveData.save_2testSolus2mat(utrue2test_numpy, unn2test_numpy, actName='utrue',
                                 actName1=R['name2act_hidden'], outPath=R['FolderName'])
    # plotData.plot_Hot_solution2test(utrue2test_numpy, size_vec2mat=size2test, actName='Utrue',
    #                                 seedNo=R['seed'], outPath=R['FolderName'])
    # plotData.plot_Hot_solution2test(unn2test_numpy, size_vec2mat=size2test, actName=R['name2act_hidden'],
    #                                 seedNo=R['seed'], outPath=R['FolderName'])

    saveData.save_test_point_wise_err2mat(point_square_error_numpy, actName=R['name2act_hidden'],
                                          outPath=R['FolderName'])

    # plotData.plot_Hot_point_wise_err(point_square_error_numpy, size_vec2mat=size2test,
    #                                  actName=R['activate_func'], seedNo=R['seed'], outPath=R['FolderName'])


if __name__ == "__main__":
    R = {}
    R['gpuNo'] = 0
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')
    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    R['model2NN'] = 'DNN'
    # R['model2NN'] = 'Scale_DNN'
    # R['model2NN'] = 'Fourier_DNN'
    # R['model2NN'] = 'Fourier_SubDNN'
    R['max_epoch'] = 50000
    # 文件保存路径设置
    if R['model2NN'] == 'Fourier_SubDNN':
        store_file = 'SoftFourierSub2Neumann_1D'
    elif R['model2NN'] == 'Fourier_DNN':
        store_file = 'SoftFourier2Neumann_1D'
    elif R['model2NN'] == 'DNN':
        store_file = 'SoftDNN2Neumann_1D'

    file2results = 'Results'
    BASE_DIR2FILE = os.path.dirname(os.path.abspath(__file__))
    split_BASE_DIR2FILE = os.path.split(BASE_DIR2FILE)
    split_BASE_DIR2FILE = os.path.split(split_BASE_DIR2FILE[0])
    BASE_DIR = split_BASE_DIR2FILE[0]
    sys.path.append(BASE_DIR)
    OUT_DIR_BASE = os.path.join(BASE_DIR, file2results)
    OUT_DIR = os.path.join(OUT_DIR_BASE, store_file)
    sys.path.append(OUT_DIR)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    current_day_time = datetime.datetime.now()  # 获取当前时间
    date_time_dir = str(current_day_time.month) + str('m_') + \
                    str(current_day_time.day) + str('d_') + str(current_day_time.hour) + str('h_') + \
                    str(current_day_time.minute) + str('m_') + str(current_day_time.second) + str('s')
    FolderName = os.path.join(OUT_DIR, date_time_dir)  # 路径连接
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    R['FolderName'] = FolderName

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  复制并保存当前文件 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if platform.system() == 'Windows':
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of multi-scale problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    R['input_dim'] = 1  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数

    R['PDE_type'] = 'PINN'
    # R['equa_name'] = '1DV_General'
    R['equa_name'] = '1DV_MultiScale'

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置(内部和边界)
    R['batch_size2interior'] = 3000  # 内部训练数据的批大小
    R['batch_size2boundary'] = 1000  # 边界训练数据的批大小
    R['batch_size2init'] = 3000
    R['batch_size2test'] = 100

    # 装载测试数据模式
    # R['testData_model'] = 'random_generate'
    # R['testData_model'] = 'generate_mesh'
    R['testData_model'] = 'load_MatData'

    R['loss_type'] = 'L2_loss'  # loss类型:L2 loss
    # R['loss_type'] = 'variational_loss'                      # loss类型:PDE变分
    # R['loss_type'] = 'lncosh_loss'
    R['lambda2lncosh'] = 0.5

    R['optimizer_name'] = 'Adam'            # 优化器
    R['learning_rate'] = 0.01            # 学习率
    # R['learning_rate'] = 0.001              # 学习率
    R['learning_rate_decay'] = 5e-5         # 学习率 decay
    R['train_model'] = 'union_training'

    # 正则化权重和偏置的模式
    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'
    # R['penalty2weight_biases'] = 0.000                    # Regularization parameter for weights
    R['penalty2weight_biases'] = 0.001  # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025                 # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    R['activate_penalty2bd_increase'] = 1
    # R['activate_penalty2bd_increase'] = 0
    R['init_boundary_penalty'] = 20

    R['activate_penalty2init_increase'] = 1
    # R['activate_penalty2init_increase'] = 0
    R['init_penalty2init'] = 20

    # 网络的频率范围设置
    R['freq'] = np.concatenate(([1], np.arange(1, 40 - 1)), axis=0)
    R['repeat_highFreq'] = True

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model2NN'] == 'Fourier_SubDNN':
        R['hidden_layers'] = (10, 20, 20, 10, 10) # （1*10+250+500+400+300+15）* 20 = 1475 *20 (subnet个数) = 29500
        # R['freq'] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.float32)
        R['freq'] = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
                             dtype=np.float32)
        # R['hidden_layers'] = (125, 200, 100, 100, 80)  # 1*125+250*200+200*200+200*100+100*100+100*50+50*1=128205
        # R['hidden_layers'] = (50, 10, 10, 10)
        # R['hidden_layers'] = (50, 80, 60, 60, 40)
    else:
        R['freq'] = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
                             dtype=np.float32)
        R['hidden_layers'] = (100, 150, 80, 80, 50)  # 1*100+100*150+150*80+80*50+50*1 = 31150
        # R['hidden_layers'] = (200, 100, 80, 50, 30)
        # R['hidden_layers'] = (250, 200, 100, 100, 80)  # 1*250+250*200+200*200+200*100+100*100+100*50+50*1=128330
        # R['hidden_layers'] = (500, 400, 300, 200, 100)
        # R['hidden_layers'] = (500, 400, 300, 300, 200, 100)

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model2NN'] == 'Fourier_SubDNN':
        R['name2act_in'] = 'sin'
        R['name2act_hidden'] = 'sin'
    elif R['model2NN'] == 'Fourier_DNN':
        R['name2act_in'] = 'sin'
        R['name2act_hidden'] = 'sin'
    elif R['model2NN'] == 'DNN':
        R['name2act_in'] = 'tanh'
        R['name2act_hidden'] = 'tanh'

    R['name2act_out'] = 'linear'
    R['sfourier'] = 1.0

    R['use_gpu'] = True

    solve_Multiscale_PDE(R)

    # R['name2act_in'] = 'relu'
    # R['name2act_in'] = 'sin'
    # R['name2act_in'] = 'tanh'
    # R['name2act_in'] = 's2relu'
    # R['name2act_in'] = 'Enh_tanh'

    # R['name2act_hidden'] = 'relu'
    # R['name2act_hidden'] = 'tanh'
    # R['name2act_hidden']' = leaky_relu'
    # R['name2act_hidden'] = 'srelu'
    # R['name2act_hidden'] = 's2relu'
    # R['name2act_hidden'] = 'sin'
    # R['name2act_hidden'] = 'sinAddcos'
    # R['name2act_hidden'] = 'Enh_tanh'
    # R['name2act_hidden'] = 'elu'
    # R['name2act_hidden'] = 'phi'
