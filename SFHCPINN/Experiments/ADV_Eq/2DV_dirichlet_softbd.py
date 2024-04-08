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
from scipy.special import erfc


# du/dt -kx * ddu/dxx - ky * ddu/dyy + vx * du/dx + vy * du/dy = f(x,y,t)
def get_fside2Smooth_Dirichlet_2d(x, y, t, alpha=0.25, beta=2.0, omega=5.0, vx=4.0, vy=4.0, kx=4.0, ky=4.0, PI=torch.pi):
    # u_true = exp(-a·t)*sin(beta*pi*x)*sin(beta*pi*y)
    exp_t = torch.exp(-alpha * t)
    du_dt = -alpha * torch.exp(-alpha * t)*torch.sin(beta*PI*x)*torch.sin(beta*PI*y)
    du_dx = torch.multiply(exp_t, beta * PI * torch.cos(beta * PI * x) * torch.sin(beta * PI * y))
    du_dy = torch.multiply(exp_t, beta * PI * torch.cos(beta * PI * y) * torch.sin(beta * PI * x))

    ddu_dxx = torch.multiply(exp_t, -1.0 * beta * beta * PI * PI * torch.sin(beta * PI * x) * torch.sin(beta * PI * y))
    ddu_dyy = torch.multiply(exp_t, -1.0 * beta * beta * PI * PI * torch.sin(beta * PI * y) * torch.sin(beta * PI * x))

    f_side = du_dt - kx*ddu_dxx - ky*ddu_dyy + vx*du_dx + vy*du_dy
    return f_side


def get_fside2Smooth_Dirichlet_2d_2(x, y, t, alpha=0.25, beta=2.0, omega=5.0, gamma=4.0, vx=4.0, vy=4.0, kx=4.0, ky=4.0,
                                    left=0.0, right=0.0, bottom=0.0, top=0.0, PI=torch.pi):
    # u_true = exp(-a·t)*(x-left)(right-x)*(y-right_bottom)*(right_top-y)
    # (x-left)(right-x) = x*right-x^2-left*right+left*x = (right-left)*x-x^2-left*right
    exp_t = torch.exp(-alpha * t)
    du_dt = -alpha * gamma * exp_t * (x-left) * (right-x) * (y-bottom) * (top-y)
    du_dx = gamma * exp_t * (right-left-2.0*x) * (y-bottom) * (top-y)
    du_dy = gamma * exp_t * (x-left) * (right-x) * (top-bottom-2.0*y)

    ddu_dxx = gamma * exp_t * (-2.0) * (y-bottom) * (top-y)
    ddu_dyy = gamma * exp_t * (x-left) * (right-x) * (-2.0)

    f_side = du_dt - kx*ddu_dxx - ky*ddu_dyy + vx*du_dx + vy*du_dy
    return f_side


def get_fside2Multiscale_Dirichlet_2d(x, y, t, alpha=0.25, beta=2.0, omega=5.0, vx=4.0, vy=4.0, kx=4.0, ky=4.0,
                                      PI=torch.pi, zeta=0.1):
    # u_true = exp(-a·t)*(sin(beta*pi*x)*sin(beta*pi*y) + zeta*sin(omega*pi*x)*sin(omega*pi*y))
    exp_t = torch.exp(-alpha * t)
    du_dt = -alpha * torch.exp(-alpha * t)*(torch.sin(beta*PI*x)*torch.sin(beta*PI*y) +
                                            zeta*torch.sin(omega*PI*y)*torch.sin(omega*PI*x))
    du_dx = torch.multiply(exp_t, beta*PI*torch.cos(beta*PI*x)*torch.sin(beta*PI*y) +
                           zeta*omega*PI*torch.cos(omega*PI*x)*torch.sin(omega*PI*y))
    du_dy = torch.multiply(exp_t, beta*PI*torch.cos(beta*PI*y)*torch.sin(beta*PI*x) +
                           zeta*omega*PI*torch.cos(omega*PI*y)*torch.sin(omega*PI*x))

    ddu_dxx = torch.multiply(exp_t, -1.0*beta*beta*PI*PI*torch.sin(beta*PI*x)*torch.sin(beta*PI*y) -
                             zeta*omega*omega*PI*PI*torch.sin(omega*PI*x)*torch.sin(omega*PI*y))
    ddu_dyy = torch.multiply(exp_t, -1.0*beta*beta*PI*PI*torch.sin(beta*PI*y)*torch.sin(beta*PI*x) -
                             zeta*omega*omega*PI*PI*torch.sin(omega*PI*y)*torch.sin(omega*PI*x))

    f_side = du_dt - kx*ddu_dxx - ky*ddu_dyy + vx*du_dx + vy*du_dy
    return f_side


def get_fside2Multiscale_Dirichlet_2d_2(x, y, t, alpha=0.25, gamma=2.0, omega=5.0, vx=4.0, vy=4.0, kx=4.0, ky=4.0,
                                        PI=torch.pi, zeta=0.1):
    # u_true = 5.0*exp(-a·t)*(x(1-x)*y(1-y) + zeta*cos(omega*pi*x)*sin(omega*pi*y))
    exp_t = torch.exp(-alpha * t)
    du_dt = -gamma * alpha * exp_t * (x*(1-x)*y*(1-y)+zeta*torch.cos(omega*PI*x)*torch.sin(omega*PI*y))
    du_dx = gamma*exp_t*((1.0 - 2*x)*y*(1-y) - zeta*omega*PI*torch.sin(omega*PI*x)*torch.sin(omega*PI*y))
    du_dy = gamma*exp_t*((1.0 - 2*y)*x*(1-x) + zeta*omega*PI*torch.cos(omega*PI*x)*torch.cos(omega*PI*y))

    ddu_dxx = gamma*exp_t*(-2.0*y*(1-y) - zeta*omega*omega*PI*PI*torch.cos(omega*PI*x)*torch.sin(omega*PI*y))
    ddu_dyy = gamma*exp_t*(-2.0*x*(1-x) - zeta*omega*omega*PI*PI*torch.cos(omega*PI*x)*torch.sin(omega*PI*y))

    f_side = du_dt - kx*ddu_dxx - ky*ddu_dyy + vx*du_dx + vy*du_dy
    return f_side


def grad_fun(model_D, XY):
    DNN = model_D(XY).reshape(-1, 1)
    X = torch.reshape(XY[:, 0], shape=[-1, 1])
    grad2DNN = torch.autograd.grad(DNN, XY, grad_outputs=torch.ones_like(X), create_graph=True, retain_graph=True)
    dDNN = grad2DNN[0]
    dDNN2x = torch.reshape(dDNN[:, 0], shape=[-1, 1])
    dDNN2y = torch.reshape(dDNN[:, 1], shape=[-1, 1])
    dDNNxxy = torch.autograd.grad(dDNN2x, XY, grad_outputs=torch.ones_like(X), create_graph=True,
                                  retain_graph=True)[0]
    dDNNxx = torch.reshape(dDNNxxy[:, 0], shape=[-1, 1])
    return DNN, dDNN2x, dDNNxx, dDNN2y


def grad_fun_nmbd(DNN, XY_bd):
    X = torch.reshape(XY_bd[:, 0], shape=[-1, 1])
    UNN = DNN(XY_bd).reshape(-1, 1)
    grad2UNN = torch.autograd.grad(UNN, XY_bd, grad_outputs=torch.ones_like(X), create_graph=True,
                                   retain_graph=True)
    dUNN = grad2UNN[0]
    dUNN2x = torch.reshape(dUNN[:, 0], shape=[-1, 1])
    return UNN, dUNN2x


def functionD(XY):
    X = torch.reshape(2 - XY[:, 0], shape=[-1, 1])
    Y = torch.reshape(100 - XY[:, 1], shape=[-1, 1])
    return torch.mul(X, Y)


def functionG(XY):
    X = torch.reshape(XY[:, 0], shape=[-1, 1])
    return torch.zeros_like(X)


def MinMaxScalling(a):
    # Min-Max scaling
    min_a = torch.min(a)
    max_a = torch.max(a)
    n2 = (a - min_a) / (max_a - min_a)
    return n2


def Eq8(z, t, p, ws, ds, split, inf, Use_torch=True):
    if Use_torch:
        tmp_t = torch.linspace(inf, t.item(), split)
        dt = (t.item() - inf) / split
        f = (z.item() - ws * tmp_t) / torch.sqrt(4 * ds * tmp_t)
        tmp_pp = (torch.exp(-(f ** 2)) / (torch.sqrt(np.pi * ds * tmp_t))) + erfc1(f) * ws * 0.5 / ds
        tmp_pp = torch.nan_to_num(tmp_pp, 0)
        out = sum(tmp_pp * p * dt)
    else:
        tmp_t = np.linspace(inf, t, split)
        dt = (t - inf) / split
        f = (z - ws * tmp_t) / np.sqrt(4 * ds * tmp_t)
        tmp_pp = (np.exp(-(f ** 2)) / (np.sqrt(np.pi * ds * tmp_t))) + erfc(f) * ws * 0.5 / ds
        tmp_pp = np.nan_to_num(tmp_pp, 0)
        out = sum(tmp_pp * p * dt)
    return out


def erfc1(x):
    tmp = x.detach().numpy()
    z = erfc(tmp)
    z = torch.from_numpy(z)
    return z


def AnalyticalSolution(z, t, p, ws, ds, split=1000, inf=-0, Use_Torch=True):
    z = z.reshape(-1, 1)
    t = t.reshape(-1, 1)
    if Use_Torch:
        N = z.shape[0]
        out = torch.zeros_like(z)
    else:
        N = z.size()
        out = np.zeros_like(z)
    for i in range(N):
        out[i] = Eq8(z[i], t[i], p, ws, ds, split, inf, Use_torch=Use_Torch)
    return out


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
        self.mat2XY = torch.tensor([[1, 0, 0],
                                    [0, 1, 0]], dtype=self.float_type, device=self.opt2device)  # 2 行 3 列
        self.mat2U = torch.tensor([[0, 0, 1]], dtype=self.float_type, device=self.opt2device)  # 1 行 3 列
        self.mat2T = torch.tensor([[0, 0, 1]], dtype=self.float_type, device=self.opt2device)  # 1 行 3 列

    def loss_it2ADVDFS(self, XY_in=None, T_in=None, fside=None, if_lambda2fside=True, loss_type='l2_losss',
                       kx=1.4, ky=1.7, vx=1, vy=1):
        assert (XY_in is not None)
        assert (T_in is not None)
        assert (fside is not None)

        shape2XY = XY_in.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)
        X = torch.reshape(XY_in[:, 0], shape=[-1, 1])
        Y = torch.reshape(XY_in[:, 1], shape=[-1, 1])

        if if_lambda2fside:
            force_side = fside(X, Y, T_in)
        else:
            force_side = fside

        XYT = torch.matmul(XY_in, self.mat2XY) + torch.matmul(T_in, self.mat2T)
        UNN = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)
        grad_UNN2XY = torch.autograd.grad(UNN, XY_in, grad_outputs=torch.ones_like(X),
                                          create_graph=True, retain_graph=True)
        dUNN2XY = grad_UNN2XY[0]

        dUNN2x = torch.reshape(dUNN2XY[:, 0], shape=[-1, 1])
        dUNN2y = torch.reshape(dUNN2XY[:, 1], shape=[-1, 1])

        grad_UNN2T = torch.autograd.grad(UNN, T_in, grad_outputs=torch.ones_like(T_in),
                                         create_graph=True, retain_graph=True)
        dUNN2t = torch.reshape(grad_UNN2T[0], shape=[-1, 1])

        dUNNxxy = torch.autograd.grad(dUNN2x, XY_in, grad_outputs=torch.ones_like(X),
                                      create_graph=True, retain_graph=True)[0]
        dUNNyxy = torch.autograd.grad(dUNN2y, XY_in, grad_outputs=torch.ones_like(X),
                                      create_graph=True, retain_graph=True)[0]

        dUNNxx = torch.reshape(dUNNxxy[:, 0], shape=[-1, 1])
        dUNNyy = torch.reshape(dUNNyxy[:, 1], shape=[-1, 1])
        # du/dt -kx * du/dxx - ky * du/dyy + vx * du/dx + vy * du/dy = f(x,y,t)
        res = dUNN2t -kx * dUNNxx - ky * dUNNyy + vx * dUNN2x + vy * dUNN2y - force_side
        if str.lower(loss_type) == 'l2_loss':
            square_loss_it = torch.mul(res, res)
            loss_it = torch.mean(square_loss_it)
        return UNN, loss_it

    def loss_in2XYT(self, XY_in=None, T_in=None, fside=None, if_lambda2fside=True, loss_type='l2_losss',
                    kx=1.4, ky=1.7, vx=1.0, vy=1.0):
        assert (XY_in is not None)
        assert (T_in is not None)
        assert (fside is not None)

        shape2XY = XY_in.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        shape2t = T_in.shape
        lenght2t_shape = len(shape2t)
        assert (lenght2t_shape == 2)
        assert (shape2t[-1] == 1)

        # 输入数据切分 + 与时间T 做结合
        X = torch.reshape(XY_in[:, 0], shape=[-1, 1])
        Y = torch.reshape(XY_in[:, 1], shape=[-1, 1])
        XYT = torch.matmul(XY_in, self.mat2XY) + torch.matmul(T_in, self.mat2T)

        # 生成对应的源项
        if if_lambda2fside:
            force_side = fside(X, Y, T_in)
        else:
            force_side = fside
        # 神经网络输出
        UNN = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)

        # dt
        grad2UNN2dt = torch.autograd.grad(UNN, T_in, grad_outputs=torch.ones_like(T_in),
                                         create_graph=True, retain_graph=True)
        dUNN2dt = grad2UNN2dt[0]

        # dx dy dxx dyy
        grad2UNN = torch.autograd.grad(UNN, XY_in, grad_outputs=torch.ones_like(X),
                                       create_graph=True, retain_graph=True)
        dUNN2xy = grad2UNN[0]
        dUNN2x = torch.reshape(dUNN2xy[:, 0], shape=[-1, 1])
        dUNN2y = torch.reshape(dUNN2xy[:, 1], shape=[-1, 1])

        dUNNxxy = torch.autograd.grad(dUNN2x, XY_in, grad_outputs=torch.ones_like(X),
                                      create_graph=True, retain_graph=True)[0]
        dUNNyxy = torch.autograd.grad(dUNN2y, XY_in, grad_outputs=torch.ones_like(X),
                                      create_graph=True, retain_graph=True)[0]

        dUNNxx = torch.reshape(dUNNxxy[:, 0], shape=[-1, 1])
        dUNNyy = torch.reshape(dUNNyxy[:, 1], shape=[-1, 1])
        # du/dt -kx * ddu/dxx - ky * ddu/dyy + vx * du/dx + vy * du/dy = f(x,y,t)
        res = dUNN2dt -kx * dUNNxx - ky * dUNNyy + vx * dUNN2x + vy * dUNN2y - force_side
        square_loss_it = torch.mul(res, res)
        loss_it = torch.mean(square_loss_it)
        return UNN, loss_it

    def loss2bd(self, XY_bd=None, T_bd=None, Ubd_exact=None, if_lambda2Ubd=True, loss_type='l2_loss', scale2lncosh=0.5):
        assert (XY_bd is not None)
        assert (Ubd_exact is not None)

        shape2XY = XY_bd.shape
        lenght2XY_shape = len(shape2XY)

        shape2T = T_bd.shape
        lenght2T_shape = len(shape2T)

        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        assert (lenght2T_shape == 2)
        assert (shape2T[-1] == 1)

        X_bd = torch.reshape(XY_bd[:, 0], shape=[-1, 1])
        Y_bd = torch.reshape(XY_bd[:, 1], shape=[-1, 1])

        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd, Y_bd, T_bd)
        else:
            Ubd = Ubd_exact

        XYT_bd = torch.matmul(XY_bd, self.mat2XY) + torch.matmul(T_bd, self.mat2T)
        UNN_bd = self.DNN(XYT_bd, scale=self.factor2freq, sFourier=self.sFourier)
        diff_bd = UNN_bd - Ubd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd = torch.mean(torch.square(diff_bd))
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_bd = (1 / scale2lncosh) * torch.mean(torch.log(torch.cosh(scale2lncosh * diff_bd)))
        return loss_bd

    def loss2init(self, XY_init=None, T_init=None, Uinit_exact=None, if_lambda2Ubd=True, loss_type='l2_loss',
                  scale2lncosh=0.5):
        assert (XY_init is not None)
        assert (Uinit_exact is not None)

        shape2XY = XY_init.shape
        lenght2XY_shape = len(shape2XY)

        shape2T = T_init.shape
        lenght2T_shape = len(shape2T)

        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        assert (lenght2T_shape == 2)
        assert (shape2T[-1] == 1)

        X_init = torch.reshape(XY_init[:, 0], shape=[-1, 1])
        Y_init = torch.reshape(XY_init[:, 1], shape=[-1, 1])

        if if_lambda2Ubd:
            Uinit = Uinit_exact(X_init, Y_init, T_init)
        else:
            Uinit = Uinit_exact

        XYT_init = torch.matmul(XY_init, self.mat2XY) + torch.matmul(T_init, self.mat2T)
        UNN_init = self.DNN(XYT_init, scale=self.factor2freq, sFourier=self.sFourier)
        diff_init = UNN_init - Uinit
        if str.lower(loss_type) == 'l2_loss':
            loss_init = torch.mean(torch.square(diff_init))
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_init = (1 / scale2lncosh) * torch.mean(torch.log(torch.cosh(scale2lncosh * diff_init)))
        return loss_init

    def loss2bd_neumann(self, XY_bd=None, Ubd_exact=None, if_lambda2Ubd=True, loss_type='l2_loss', scale2lncosh=0.5):
        assert (XY_bd is not None)
        assert (Ubd_exact is not None)

        shape2XY = XY_bd.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)
        X = torch.reshape(XY_bd[:, 0], shape=[-1, 1])
        Y = torch.reshape(XY_bd[:, 1], shape=[-1, 1])
        if if_lambda2Ubd:
            U_bd = Ubd_exact(X, Y)
        else:
            U_bd = Ubd_exact

        UNN = self.DNN(XY_bd, scale=self.factor2freq, sFourier=self.sFourier)
        grad2UNN = torch.autograd.grad(UNN, XY_bd, grad_outputs=torch.ones_like(X), create_graph=True,
                                       retain_graph=True)
        dUNN = grad2UNN[0]
        dUNN2x = torch.reshape(dUNN[:, 0], shape=[-1, 1])
        diff_bd = dUNN2x - U_bd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd = torch.mean(torch.square(diff_bd))
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_bd = (1 / scale2lncosh) * torch.mean(torch.log(torch.cosh(scale2lncosh * diff_bd)))
        return loss_bd

    def get_regularSum2WB(self):
        sum2WB = self.DNN.get_regular_sum2WB(self.opt2regular_WB)
        return sum2WB

    def evalue_MscaleDNN(self, XY_points=None, T_points=None ):
        assert (XY_points is not None)
        assert (T_points is not None)
        shape2XY = XY_points.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        shape2T = T_points.shape
        lenght2T_shape = len(shape2T)
        assert (lenght2T_shape == 2)
        assert (shape2T[-1] == 1)

        XYT = torch.matmul(XY_points, self.mat2XY) + torch.matmul(T_points, self.mat2T)
        UNN = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)
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
    batchsize_init = R['batch_size2init']

    bd_penalty_init = R['init_boundary_penalty']  # Regularization parameter for boundary conditions
    penalty2WB = R['penalty2weight_biases']       # Regularization parameter for weights and biases
    learning_rate = R['learning_rate']

    if R['equa_name'] == 'Smooth_Dirichlet':
        region_l = 0.0
        region_r = 1.0
        region_b = 0.0
        region_t = 1.0
        init_time = 0.0
        end_time = 5.0
        vx = 4.0     # 一次项
        vy = 4.0     # 一次项
        kx = 1.0     # 二次项
        ky = 1.0     # 二次项
        pi = np.pi
        alpha = 0.25
        beta = 2.0
        omega = 5.0
        u_true = lambda x, y, t: torch.mul(torch.exp(-alpha*t), torch.mul(torch.sin(beta*pi*x), torch.sin(beta*pi*y)))
        u_init = lambda x, y, t: torch.mul(torch.exp(-alpha*t), torch.mul(torch.sin(beta*pi*x), torch.sin(beta*pi*y)))
        u_left = lambda x, y, t: torch.mul(torch.exp(-alpha*t), torch.mul(torch.sin(beta*pi*x), torch.sin(beta*pi*y)))
        u_right = lambda x, y, t: torch.mul(torch.exp(-alpha*t), torch.mul(torch.sin(beta*pi*x), torch.sin(beta*pi*y)))
        u_bottom = lambda x, y, t: torch.mul(torch.exp(-alpha*t), torch.mul(torch.sin(beta*pi*x), torch.sin(beta*pi*y)))
        u_top = lambda x, y, t: torch.mul(torch.exp(-alpha*t), torch.mul(torch.sin(beta*pi*x), torch.sin(beta*pi*y)))

        f_side = lambda x, y, t: get_fside2Smooth_Dirichlet_2d(x, y, t, alpha=alpha, beta=beta, omega=5.0, vx=vx, vy=vy,
                                                               kx=kx, ky=ky, PI=torch.pi)
    elif R['equa_name'] == 'Smooth_Dirichlet2':
        region_l = 0.0
        region_r = 4.0
        region_b = 0.0
        region_t = 4.0
        init_time = 0.0
        end_time = 5.0
        vx = 4.0     # 一次项
        vy = 4.0     # 一次项
        kx = 1.0     # 二次项
        ky = 1.0     # 二次项
        pi = np.pi
        alpha = 0.25
        beta = 2.0
        omega = 4.0
        gamma = 1.0
        u_true = lambda x, y, t: gamma*torch.exp(-alpha*t)*(x-region_l)*(region_r-x)*(y-region_b)*(region_t-y)
        u_init = lambda x, y, t: gamma*torch.exp(-alpha*t)*(x-region_l)*(region_r-x)*(y-region_b)*(region_t-y)
        u_left = lambda x, y, t: gamma*torch.exp(-alpha*t)*(x-region_l)*(region_r-x)*(y-region_b)*(region_t-y)
        u_right = lambda x, y, t: gamma*torch.exp(-alpha*t)*(x-region_l)*(region_r-x)*(y-region_b)*(region_t-y)
        u_bottom = lambda x, y, t: gamma*torch.exp(-alpha*t)*(x-region_l)*(region_r-x)*(y-region_b)*(region_t-y)
        u_top = lambda x, y, t: gamma*torch.exp(-alpha*t)*(x-region_l)*(region_r-x)*(y-region_b)*(region_t-y)

        f_side = lambda x, y, t: get_fside2Smooth_Dirichlet_2d_2(
            x, y, t, alpha=alpha, beta=beta, omega=5.0, gamma=gamma, vx=vx, vy=vy, kx=kx, ky=ky, left=region_l, right=region_r,
            bottom=region_b, top=region_t, PI=torch.pi)
    elif R['equa_name'] == 'Multiscale_Dirichlet':
        region_l = 0.0
        region_r = 1.0
        region_b = 0.0
        region_t = 1.0
        init_time = 0.0
        end_time = 5.0
        vx = 4.0  # 一次项
        vy = 4.0  # 一次项
        kx = 1.0  # 二次项
        ky = 1.0  # 二次项
        pi = np.pi
        alpha = 0.25
        beta = 2.0
        omega = 10.0
        zeta = 0.1

        u_true = lambda x, y, t: torch.mul(torch.exp(-alpha*t), torch.mul(torch.sin(beta*pi*x), torch.sin(beta*pi*y)) +
                                           zeta*torch.mul(torch.sin(omega*pi*x), torch.sin(omega*pi*y)))
        u_init = lambda x, y, t: torch.mul(torch.exp(-alpha*t), torch.mul(torch.sin(beta*pi*x), torch.sin(beta*pi*y)) +
                                           zeta*torch.mul(torch.sin(omega*pi*x), torch.sin(omega*pi*y)))
        u_left = lambda x, y, t: torch.mul(torch.exp(-alpha*t), torch.mul(torch.sin(beta*pi*x), torch.sin(beta*pi*y)) +
                                           zeta*torch.mul(torch.sin(omega*pi*x), torch.sin(omega*pi*y)))
        u_right = lambda x, y, t: torch.mul(torch.exp(-alpha*t), torch.mul(torch.sin(beta*pi*x), torch.sin(beta*pi*y)) +
                                           zeta*torch.mul(torch.sin(omega*pi*x), torch.sin(omega*pi*y)))
        u_bottom = lambda x, y, t: torch.mul(torch.exp(-alpha*t), torch.mul(torch.sin(beta*pi*x), torch.sin(beta*pi*y)) +
                                           zeta*torch.mul(torch.sin(omega*pi*x), torch.sin(omega*pi*y)))
        u_top = lambda x, y, t: torch.mul(torch.exp(-alpha*t), torch.mul(torch.sin(beta*pi*x), torch.sin(beta*pi*y)) +
                                           zeta*torch.mul(torch.sin(omega*pi*x), torch.sin(omega*pi*y)))
        f_side = lambda x, y, t: get_fside2Multiscale_Dirichlet_2d(x, y, t, alpha=alpha, beta=beta, omega=omega, vx=vx,
                                                                   vy=vy, kx=kx, ky=ky, PI=torch.pi, zeta=zeta)
    else:
        region_l = 0.0
        region_r = 1.0
        region_b = 0.0
        region_t = 1.0
        init_time = 0.0
        end_time = 5.0
        vx = 4.0  # 一次项
        vy = 4.0  # 一次项
        kx = 1.0  # 二次项
        ky = 1.0  # 二次项
        pi = np.pi
        alpha = 0.25
        gamma = 2.0
        omega = 5.0
        zeta = 0.1

        u_true = lambda x, y, t: gamma*torch.exp(-alpha * t)*(
                x*(1-x)*y*(1-y) + zeta * torch.cos(omega * pi * x)*torch.sin(omega * pi * y))
        u_init = lambda x, y, t: gamma*torch.exp(-alpha * t)*(
                x*(1-x)*y*(1-y) + zeta * torch.cos(omega * pi * x)*torch.sin(omega * pi * y))
        u_left = lambda x, y, t: gamma*torch.exp(-alpha * t)*(
                x*(1-x)*y*(1-y) + zeta * torch.cos(omega * pi * x)*torch.sin(omega * pi * y))
        u_right = lambda x, y, t: gamma*torch.exp(-alpha * t)*(
                x*(1-x)*y*(1-y) + zeta * torch.cos(omega * pi * x)*torch.sin(omega * pi * y))
        u_bottom = lambda x, y, t: gamma*torch.exp(-alpha * t)*(
                x*(1-x)*y*(1-y) + zeta * torch.cos(omega * pi * x)*torch.sin(omega * pi * y))
        u_top = lambda x, y, t: gamma*torch.exp(-alpha * t)*(
                x*(1-x)*y*(1-y) + zeta * torch.cos(omega * pi * x)*torch.sin(omega * pi * y))
        f_side = lambda x, y, t: get_fside2Multiscale_Dirichlet_2d_2(
            x, y, t, alpha=alpha, gamma=gamma, omega=omega, vx=vx, vy=vy, kx=kx, ky=ky, PI=torch.pi, zeta=zeta)

    mscalednn = SoftPINN(input_dim=R['input_dim'], out_dim=R['output_dim'], hidden_layer=R['hidden_layers'],
                         Model_name=R['model2NN'], name2actIn=R['name2act_in'], name2actHidden=R['name2act_hidden'],
                         name2actOut=R['name2act_out'], opt2regular_WB='L0', repeat_highFreq=R['repeat_highFreq'],
                         type2numeric='float32', factor2freq=R['freq'], use_gpu=R['use_gpu'], No2GPU=R['gpuNo'])
    if True == R['use_gpu']:
        mscalednn = mscalednn.cuda(device='cuda:' + str(R['gpuNo']))

    params2Net = mscalednn.DNN.parameters()

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
    loss_init_all = []

    if R['testData_model'] == 'random_generate':
        test_bach_size = 4000
        random_coord2x = (region_r - region_l) * np.random.rand(test_bach_size, 1) + region_l
        random_coord2y = (region_t - region_b) * np.random.rand(test_bach_size, 1) + region_b

        test_xy_bach = np.concatenate([random_coord2x, random_coord2y], axis=-1, dtype=np.float32)
        test_time_batch = 0.5 * (init_time + end_time) * np.ones(shape=[test_bach_size, 1], dtype=np.float32)
    elif R['testData_model'] == 'load_RegularDomain_Data':
        size2test = 258
        mat_data_path = '../data2RegularDomain_2D'
        test_xy_bach = get_meshData2Advection(data_path=mat_data_path,  mesh_number=7, to_torch=False, to_float=True,
                                              to_cuda=False, gpu_no=0, use_grad2x=False)
        shape2xy = np.shape(test_xy_bach)
        batch2test = shape2xy[0]
        test_time_batch = 0.5 * np.ones(shape=[batch2test, 1], dtype=np.float32)
    else:
        if R['equa_name'] == 'Smooth_Dirichlet2':
            test_xy_bach = dataUtilizer2torch.load_data2porous_domain(
                region_left=region_l, region_right=region_r, region_bottom=region_b, region_top=region_t,
                to_torch=False, to_float=True, to_cuda=False, gpu_no=0, use_grad2x=False, scale_trans=True,
                scale2x=region_r-region_l, scale2y=region_t-region_b, base2x=region_l, base2y=region_b)
        else:
            test_xy_bach = dataUtilizer2torch.load_data2porous_domain(
                region_left=region_l, region_right=region_r, region_bottom=region_b, region_top=region_t,
                to_torch=False, to_float=True, to_cuda=False, gpu_no=0, use_grad2x=False)
        shape2xy = np.shape(test_xy_bach)
        batch2test = shape2xy[0]
        test_time_batch = 0.5*(init_time+end_time) * np.ones(shape=[batch2test, 1], dtype=np.float32)

    saveData.save_testData_or_solus2mat(test_xy_bach, dataName='testXY', outPath=R['FolderName'])

    test_xy_torch = torch.from_numpy(test_xy_bach)
    test_time_torch = torch.from_numpy(test_time_batch)
    if True == R['use_gpu']:
        test_xy_torch = test_xy_torch.cuda(device='cuda:' + str(R['gpuNo']))
        test_time_torch = test_time_torch.cuda(device='cuda:' + str(R['gpuNo']))

    # 生成test data的真实值
    Utrue2test = u_true(torch.reshape(test_xy_torch[:, 0], shape=[-1, 1]),
                        torch.reshape(test_xy_torch[:, 1], shape=[-1, 1]),
                        torch.reshape(test_time_torch, shape=[-1, 1]))

    t_init_batch = torch.zeros([batchsize_init, 1])
    if True == R['use_gpu']:
        t_init_batch = t_init_batch.cuda(device='cuda:' + str(R['gpuNo']))

    t0 = time.time()
    for i_epoch in range(R['max_epoch'] + 1):
        xy_it_batch = dataUtilizer2torch.rand_in_2D(
            batch_size=batchsize_it, variable_dim=R['input_dim'] - 1, region_left=region_l, region_right=region_r,
            region_bottom=region_b, region_top=region_t, to_torch=True, to_float=True, to_cuda=R['use_gpu'],
            gpu_no=0, use_grad2x=True)

        t_it_batch = dataUtilizer2torch.rand_in_1D(
            batch_size=batchsize_it, variable_dim=1, region_a=init_time, region_b=end_time, to_torch=True,
            to_float=True, to_cuda=R['use_gpu'], gpu_no=0, use_grad2x=True)

        t_bd_batch = dataUtilizer2torch.rand_in_1D(
            batch_size=batchsize_bd, variable_dim=1, region_a=init_time, region_b=end_time, to_torch=True,
            to_float=True, to_cuda=R['use_gpu'], gpu_no=0, use_grad2x=False)

        xy_left_bd, xy_right_bd, xy_bottom_bd, xy_top_bd = dataUtilizer2torch.rand_bd_2D(
            batch_size=batchsize_bd, variable_dim=R['input_dim']-1, region_left=region_l, region_right=region_r,
            region_bottom=region_b, region_top=region_t, to_torch=True, to_float=True, to_cuda=R['use_gpu'],
            use_grad=False, gpu_no=0)

        xy_init_batch = dataUtilizer2torch.rand_in_2D(
            batch_size=batchsize_init, variable_dim=R['input_dim'] - 1, region_left=region_l, region_right=region_r,
            region_bottom=region_b, region_top=region_t, to_torch=True, to_float=True, to_cuda=R['use_gpu'],
            gpu_no=0, use_grad2x=False)

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

        UNN2train, loss_it = mscalednn.loss_in2XYT(XY_in=xy_it_batch, T_in=t_it_batch, loss_type=R['loss_type'], kx=kx,
                                                   ky=ky, vx=vx, vy=vy, fside=f_side)

        loss_bd2left = mscalednn.loss2bd(XY_bd=xy_left_bd, T_bd=t_bd_batch,
                                         Ubd_exact=u_left, loss_type=R['loss_type'])

        loss_bd2right = mscalednn.loss2bd(XY_bd=xy_right_bd, T_bd=t_bd_batch,
                                          Ubd_exact=u_right, loss_type=R['loss_type'])

        loss_bd2bottom = mscalednn.loss2bd(XY_bd=xy_bottom_bd, T_bd=t_bd_batch,
                                           Ubd_exact=u_bottom, loss_type=R['loss_type'])

        loss_bd2top = mscalednn.loss2bd(XY_bd=xy_top_bd, T_bd=t_bd_batch,
                                        Ubd_exact=u_top, loss_type=R['loss_type'])

        loss_init = mscalednn.loss2init(XY_init=xy_init_batch, T_init=t_init_batch,
                                        Uinit_exact=u_init, loss_type=R['loss_type'])

        loss_bd = loss_bd2left + loss_bd2right + loss_bd2bottom + loss_bd2top
        PWB = mscalednn.get_regularSum2WB()
        loss = loss_it + temp_penalty_bd * loss_bd + loss_init * bd_penalty_init
        Uexact2train = u_true(torch.reshape(xy_it_batch[:, 0], shape=[-1, 1]),
                              torch.reshape(xy_it_batch[:, 1], shape=[-1, 1]),
                              torch.reshape(t_it_batch, shape=[-1, 1]))

        loss_it_all.append(loss_it.item())
        loss_bd_all.append(loss_bd.item())
        loss_all.append(loss.item())

        optimizer.zero_grad()  # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()        # 对loss关于Ws和Bs求偏导
        optimizer.step()       # 更新参数Ws和Bs
        scheduler.step()

        train_mse = torch.mean(torch.square(UNN2train - Uexact2train))
        train_rel = train_mse / torch.mean(torch.square(Uexact2train))

        train_mse_all.append(train_mse.item())
        train_rel_all.append(train_rel.item())

        if i_epoch % 1000 == 0:
            run_times = time.time() - t0
            tmp_lr = optimizer.param_groups[0]['lr']
            DNN_tools.print_and_log_train_one_epoch_pinn2hard(i_epoch, run_times, tmp_lr, loss_it.item(),
                                                              loss_bd.item(),
                                                              loss.item(), train_mse.item(), train_rel.item(),
                                                              log_out=log_fileout)

            test_epoch.append(i_epoch / 1000)

            UNN2test = mscalednn.evalue_MscaleDNN(XY_points=test_xy_torch, T_points=test_time_torch)

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

    plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_in', outPath=R['FolderName'], yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', outPath=R['FolderName'], yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', outPath=R['FolderName'], yaxis_scale=True)

    plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=R['name2act_hidden'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    # ----------------------  save testing results to mat files, then plot them --------------------------------
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

    saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=R['name2act_hidden'], outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=R['name2act_hidden'],
                              outPath=R['FolderName'], yaxis_scale=True)

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
        store_file = 'SoftFourierSub2Dirichlet_2D'
    if R['model2NN'] == 'Fourier_DNN':
        store_file = 'SoftFourier2Dirichlet_2D'
    elif R['model2NN'] == 'DNN':
        store_file = 'SoftDNN2Dirichlet_2D'

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
    R['input_dim'] = 3   # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数

    R['PDE_type'] = 'SoftPINN'
    # R['equa_name'] = 'Smooth_Dirichlet'
    R['equa_name'] = 'Smooth_Dirichlet2'
    # R['equa_name'] = 'Multiscale_Dirichlet'
    # R['equa_name'] = 'Multiscale_Dirichlet2'

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置(内部和边界)
    R['batch_size2interior'] = 5000  # 内部训练数据的批大小
    R['batch_size2boundary'] = 500  # 边界训练数据的批大小
    R['batch_size2init'] = 3000
    R['batch_size2test'] = 100

    # 装载测试数据模式
    R['testData_model'] = 'load_porous_Data'
    # R['testData_model'] = 'load_RegularDomain_Data'
    # R['testData_model'] = 'random_generate'

    R['loss_type'] = 'L2_loss'         # loss类型:L2 loss
    # R['loss_type'] = 'lncosh_loss'
    R['lambda2lncosh'] = 0.5

    R['optimizer_name'] = 'Adam'            # 优化器
    R['learning_rate'] = 0.01               # 学习率
    # R['learning_rate'] = 0.001            # 学习率
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
    R['repeat_highFreq'] = False

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model2NN'] == 'Fourier_SubDNN':
        R['hidden_layers'] = (10, 20, 20, 20, 10)  # （1*10+250+500+400+300+15）* 20 = 1475 *20 (subnet个数) = 29500
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
    if R['model2NN'] == 'Fourier_DNN':
        R['name2act_in'] = 'sin'
        R['name2act_hidden'] = 'sin'
    elif R['model2NN'] == 'DNN':
        R['name2act_in'] = 'tanh'
        R['name2act_hidden'] = 'tanh'

    R['name2act_out'] = 'linear'
    R['sfourier'] = 1.0

    R['use_gpu'] = True
    # R['use_gpu'] = False

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
