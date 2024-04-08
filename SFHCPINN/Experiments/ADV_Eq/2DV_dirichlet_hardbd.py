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


def temp_func(z):
    return torch.mul(z, 1-z)


# du/dt -kx * ddu/dxx - ky * ddu/dyy + vx * du/dx + vy * du/dy = f(x,y,t)
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


class HardPINN(tn.Module):
    def __init__(self, input_dim=4, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='relu',
                 name2actHidden='relu', name2actOut='linear', opt2regular_WB='L2', type2numeric='float32',
                 factor2freq=None, sFourier=1.0, use_gpu=False, No2GPU=0, repeat_highFreq=True):
        super(HardPINN, self).__init__()
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

    def loss_in(self, XY=None, t=None, fside=None, if_lambda2fside=True, loss_type='l2_losss', kx=1.0, ky=1.0,
                        vx=4.0, vy=4.0, model2Dist=None, model2Gene=None, if_NN2model_D=True, if_NN2model_G=True):
        '''
        Args:
            XY: 输入的内部点
            t: 时间
            fside: 原函数
            if_lambda2fside: Bool：原函数为Lambda函数
            loss_type: 损失函数类型
            kx: 二阶导的系数
            ky: 二阶导的系数
            vx: 一阶导的系数
            vy: 一阶导的系数
            model_D: 距离函数
            model_G: 拓展函数

        Returns:
            LNN : 预测值
            loss : 损失
        '''
        assert (XY is not None)
        assert (t is not None)
        assert (fside is not None)

        shape2XY = XY.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        shape2t = t.shape
        lenght2t_shape = len(shape2t)
        assert (lenght2t_shape == 2)
        assert (shape2t[-1] == 1)

        # 输入数据切分 + 与时间T 做结合
        X = torch.reshape(XY[:, 0], shape=[-1, 1])
        Y = torch.reshape(XY[:, 1], shape=[-1, 1])

        XYT = torch.matmul(XY, self.mat2XY) + torch.matmul(t, self.mat2T)

        # 生成对应的源项
        if if_lambda2fside:
            force_side = fside(X, Y, t)
        else:
            force_side = fside
        # 神经网络输出
        UNN = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)

        # dt
        grad2UNN2t = torch.autograd.grad(UNN, t, grad_outputs=torch.ones_like(t),
                                         create_graph=True, retain_graph=True)
        dUNN2t = grad2UNN2t[0]

        # dx dy dxx dyy
        grad2UNN = torch.autograd.grad(UNN, XY, grad_outputs=torch.ones_like(X),
                                       create_graph=True, retain_graph=True)
        dUNN = grad2UNN[0]
        dUNN2x = torch.reshape(dUNN[:, 0], shape=[-1, 1])
        dUNN2y = torch.reshape(dUNN[:, 1], shape=[-1, 1])

        dUNNxxy = torch.autograd.grad(dUNN2x, XY, grad_outputs=torch.ones_like(X),
                                      create_graph=True, retain_graph=True)[0]
        dUNNyxy = torch.autograd.grad(dUNN2y, XY, grad_outputs=torch.ones_like(X),
                                      create_graph=True, retain_graph=True)[0]

        dUNNxx = torch.reshape(dUNNxxy[:, 0], shape=[-1, 1])
        dUNNyy = torch.reshape(dUNNyxy[:, 1], shape=[-1, 1])

        # 神经网络输出
        if if_NN2model_D:
            DNN = model2Dist(XYT)
        else:
            DNN = model2Dist(X, Y, t)

        # dt
        grad2DNN2t = torch.autograd.grad(DNN, t, grad_outputs=torch.ones_like(t), create_graph=True, retain_graph=True)
        dDNN2t = grad2DNN2t[0]
        # dx dy dxx dyy
        grad2DNN = torch.autograd.grad(DNN, XY, grad_outputs=torch.ones_like(X), create_graph=True, retain_graph=True)
        dDNN = grad2DNN[0]
        dDNN2x = torch.reshape(dDNN[:, 0], shape=[-1, 1])
        dDNN2y = torch.reshape(dDNN[:, 1], shape=[-1, 1])

        dDNNxxy = torch.autograd.grad(dDNN2x, XY, grad_outputs=torch.ones_like(X),
                                      create_graph=True, retain_graph=True)[0]
        dDNNyxy = torch.autograd.grad(dDNN2y, XY, grad_outputs=torch.ones_like(X),
                                      create_graph=True, retain_graph=True)[0]

        dDNNxx = torch.reshape(dDNNxxy[:, 0], shape=[-1, 1])
        dDNNyy = torch.reshape(dDNNyxy[:, 1], shape=[-1, 1])

        if if_NN2model_G:
            GNN = model2Gene(XYT)
        else:
            GNN = model2Gene(X, Y, t)
        # dt
        grad2GNN2t = torch.autograd.grad(GNN, t, grad_outputs=torch.ones_like(t), create_graph=True, retain_graph=True,
                                         allow_unused=True)
        dGNN2t = grad2GNN2t[0]

        # dx dy dxx dyy
        grad2GNN = torch.autograd.grad(GNN, XY, grad_outputs=torch.ones_like(X), create_graph=True, retain_graph=True)
        dGNN = grad2GNN[0]
        dGNN2x = torch.reshape(dGNN[:, 0], shape=[-1, 1])
        dGNN2y = torch.reshape(dGNN[:, 1], shape=[-1, 1])

        dGNNxxy = torch.autograd.grad(dGNN2x, XY, grad_outputs=torch.ones_like(X),
                                      create_graph=True, retain_graph=True)[0]
        dGNNyxy = torch.autograd.grad(dGNN2y, XY, grad_outputs=torch.ones_like(X),
                                      create_graph=True, retain_graph=True)[0]

        dGNNxx = torch.reshape(dGNNxxy[:, 0], shape=[-1, 1])
        dGNNyy = torch.reshape(dGNNyxy[:, 1], shape=[-1, 1])

        # du/dt -kx * ddu/dxx - ky * ddu/dyy + vx * du/dx + vy * du/dy = f(x,y,t)
        DT = torch.mul(DNN, dUNN2t) + torch.mul(dDNN2t, UNN)
        DX = dGNN2x + torch.mul(DNN, dUNN2x) + torch.mul(dDNN2x, UNN)
        DY = dGNN2y + torch.mul(DNN, dUNN2y) + torch.mul(dDNN2y, UNN)
        DXX = dGNNxx + torch.mul(DNN, dUNNxx) + 2 * torch.mul(dDNN2x, dUNN2x) + torch.mul(dDNNxx, UNN)
        DYY = dGNNyy + torch.mul(DNN, dUNNyy) + 2 * torch.mul(dDNN2y, dUNN2y) + torch.mul(dDNNyy, UNN)
        res = DT + vx * DX + vy * DY - kx * DXX - ky * DYY - force_side
        LNN = GNN + torch.mul(DNN, UNN)
        if str.lower(loss_type) == 'ritz_loss' or str.lower(loss_type) == 'variational_loss':
            dUNN_2Norm = torch.reshape(torch.sum(torch.mul(dUNN, dUNN), dim=-1), shape=[-1, 1])  # 按行求和
            loss_it_ritz = (1.0 / 2) * dUNN_2Norm - torch.mul(torch.reshape(force_side, shape=[-1, 1]), UNN)
            loss_it = torch.mean(loss_it_ritz)
        elif str.lower(loss_type) == 'l2_loss':
            square_loss_it = torch.mul(res, res)
            loss_it = torch.mean(square_loss_it)
        return LNN, loss_it

    def loss_init(self, XY_init=None, t_init=None, Uinit_exact=None, if_lambda2Uinit=True, loss_type='l2_loss',
                  scale2lncosh=0.5, model2Dist=None, model2Gene=None, if_NN2model_D=True, if_NN2model_G=True):
        '''

        Args:
            XY_init: t=init_t时， xy的取点
            t_init: t= init_t
            Uinit_exact: 初始边界的函数
            if_lambda2Uinit: 初始边界函数g是否为lambda函数
            loss_type:
            scale2lncosh:

        Returns: 初始点的损失

        '''
        assert (XY_init is not None)
        assert (t_init is not None)
        assert (Uinit_exact is not None)

        shape2XY = XY_init.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        shape2t = t_init.shape
        lenght2t_shape = len(shape2t)
        assert (lenght2t_shape == 2)
        assert (shape2t[-1] == 1)

        X = torch.reshape(XY_init[:, 0], shape=[-1, 1])
        Y = torch.reshape(XY_init[:, 1], shape=[-1, 1])
        # T_init = torch.reshape(t_init, shape=[-1, 1])

        if if_lambda2Uinit:
            Linit = Uinit_exact(X, Y)
        else:
            Linit = Uinit_exact

        XYT = torch.matmul(XY_init, self.mat2XY) + torch.matmul(t_init, self.mat2T)
        UNN_init = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)

        # 神经网络输出
        if if_NN2model_D:
            DNN_init = model2Dist(XYT)
        else:
            DNN_init = model2Dist(X, Y, t_init)

        if if_NN2model_G:
            GNN_init = model2Gene(XYT)
        else:
            GNN_init = model2Gene(X, Y, t_init)

        LNN_init = GNN_init + torch.mul(DNN_init, UNN_init)

        diif_init = LNN_init - Linit

        if str.lower(loss_type) == 'l2_loss':
            loss_init_square = torch.square(diif_init)
            loss_init = torch.mean(loss_init_square)
            return loss_init
        elif str.lower(loss_type) == 'lncosh' or str.lower(loss_type) == 'lncosh_loss':
            loss_init_lncosh = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * diif_init))
            loss_init = torch.mean(loss_init_lncosh)
            return loss_init

    def loss_bd2dirichlet(self, XY_bd=None, t_bd=None, Ubd_exact=None, if_lambda2Ubd=True, loss_type='l2_loss',
                          scale2lncosh=0.5, model2Dist=None, model2Gene=None, if_NN2model_D=True, if_NN2model_G=True):
        assert (XY_bd is not None)
        assert (t_bd is not None)
        assert (Ubd_exact is not None)

        shape2XY = XY_bd.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        shape2t = t_bd.shape
        lenght2t_shape = len(shape2t)
        assert (lenght2t_shape == 2)
        assert (shape2t[-1] == 1)

        X_bd = torch.reshape(XY_bd[:, 0], shape=[-1, 1])
        Y_bd = torch.reshape(XY_bd[:, 1], shape=[-1, 1])

        if if_lambda2Ubd:
            Lbd = Ubd_exact(X_bd, Y_bd, t_bd)
        else:
            Lbd = Ubd_exact

        XYT = torch.matmul(XY_bd, self.mat2XY) + torch.matmul(t_bd, self.mat2T)
        UNN_bd = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)

        # 神经网络输出
        if if_NN2model_D:
            DNN_bd = model2Dist(XYT)
        else:
            DNN_bd = model2Dist(X_bd, Y_bd, t_bd)

        if if_NN2model_G:
            GNN_bd = model2Gene(XYT)
        else:
            GNN_bd = model2Gene(X_bd, Y_bd, t_bd)

        LNN_bd = GNN_bd + torch.mul(DNN_bd, UNN_bd)

        diff_bd = LNN_bd - Lbd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd_square = torch.square(diff_bd)
            loss_bd = torch.mean(loss_bd_square)
            return loss_bd
        elif str.lower(loss_type) == 'lncosh' or str.lower(loss_type) == 'lncosh_loss':
            loss_bd_lncosh = (1 / scale2lncosh) * torch.log(torch.cosh(scale2lncosh * diff_bd))
            loss_bd = torch.mean(loss_bd_lncosh)
            return loss_bd

    def loss2bd_neumann(self, XY_bd=None, T_bd=None, Ubd_exact=None, if_lambda2Ubd=True, loss_type='l2_loss',
                        scale2lncosh=0.5, model2Dist=None, model2Gene=None, if_NN2model_D=True, if_NN2model_G=True):
        assert (XY_bd is not None)
        assert (Ubd_exact is not None)

        shape2XY = XY_bd.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)
        X_bd = torch.reshape(XY_bd[:, 0], shape=[-1, 1])
        Y_bd = torch.reshape(XY_bd[:, 1], shape=[-1, 1])
        T_bd = torch.reshape(T_bd, shape=[-1, 1])
        if if_lambda2Ubd:
            U_bd = Ubd_exact(X_bd, Y_bd)
        else:
            U_bd = Ubd_exact

        XYT = torch.matmul(XY_bd, self.mat2XY) + torch.matmul(T_bd, self.mat2T)
        UNN_bd = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)

        # 神经网络输出
        if if_NN2model_D:
            DNN_bd = model2Dist(XYT)
        else:
            DNN_bd = model2Dist(X_bd, Y_bd, T_bd)

        if if_NN2model_G:
            GNN_bd = model2Gene(XYT)
        else:
            GNN_bd = model2Gene(X_bd, Y_bd, T_bd)

        LNN_bd = GNN_bd + torch.mul(DNN_bd, UNN_bd)

        grad2LNN_bd = torch.autograd.grad(LNN_bd, XY_bd, grad_outputs=torch.ones_like(X_bd), create_graph=True,
                                          retain_graph=True)

        dLNN = grad2LNN_bd[0]
        dLNN2x = torch.reshape(dLNN[:, 0], shape=[-1, 1])
        diff_bd = dLNN2x - U_bd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd = torch.mean(torch.square(diff_bd))
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_bd = (1 / scale2lncosh) * torch.mean(torch.log(torch.cosh(scale2lncosh * diff_bd)))
        return loss_bd

    def get_regularSum2WB(self):
        sum2WB = self.DNN.get_regular_sum2WB(self.opt2regular_WB)
        return sum2WB

    def eval_MscaleDNN(self, XY_points=None, t_points=None, model2Dist=None, model2Gene=None, if_NN2model_D=True,
                         if_NN2model_G=True):
        assert (XY_points is not None)
        shape2XY = XY_points.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        X = torch.reshape(XY_points[:, 0], shape=[-1, 1])
        Y = torch.reshape(XY_points[:, 1], shape=[-1, 1])

        XYT = torch.matmul(XY_points, self.mat2XY) + torch.matmul(t_points, self.mat2T)
        UNN = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)

        # 神经网络输出
        if if_NN2model_D:
            DNN = model2Dist(XYT)
        else:
            DNN = model2Dist(X, Y, t_points)

        if if_NN2model_G:
            GNN = model2Gene(XYT)
        else:
            GNN = model2Gene(X, Y, t_points)

        LNN = GNN + torch.mul(DNN, UNN)

        return LNN


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']  # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)  # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2train', R['name2act_hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Log_Print.dictionary_out2file(R, log_fileout)

    # 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_init = R['batch_size2init']

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
        f_side = lambda x, y, t: get_fside2Smooth_Dirichlet_2d(x, y, t, alpha=alpha, beta=beta, omega=omega, vx=vx, vy=vy,
                                                               kx=kx, ky=ky, PI=torch.pi)

        model_G = lambda x, y, t: torch.mul(torch.sin(beta * pi * x), torch.sin(beta * pi * y))

        model_D = lambda x, y, t: torch.mul(torch.mul(temp_func(x), temp_func(y)), t / end_time)

        # temp2 = lambda x: torch.ones_like(x) - torch.exp(-x)
        # temp3 = lambda x, y: torch.mul(temp2(x), temp2(y))
        # model_D = lambda x, y, t: torch.mul(temp3(x, y), temp2(t))
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

        # model_D = lambda x, y, t: (t/end_time)*(x-region_l)*(region_r-x)*(y-region_b)*\
        #                           (region_t-y)/((region_r-region_l)*(region_t-region_b))

        model_D = lambda x, y, t: (t/end_time)*torch.sin(0.25*torch.pi*x)*torch.sin(0.25*torch.pi*y)

        model_G = lambda x, y, t: gamma * (x - region_l) * (region_r - x) * (y - region_b) * (region_t - y)

        # model_G = lambda x, y, t: gamma * torch.sin(0.25*torch.pi*x)*torch.sin(0.25*torch.pi*y)
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
        f_side = lambda x, y, t: get_fside2Multiscale_Dirichlet_2d(x, y, t, alpha=alpha, beta=beta, omega=omega, vx=vx,
                                                                   vy=vy, kx=kx, ky=ky, PI=torch.pi, zeta=zeta)

        # model_G = lambda x, y, t: torch.mul(torch.sin(pi * x), torch.sin(pi * y))

        # model_G = lambda x, y, t: torch.mul(torch.sin(beta*pi * x), torch.sin(beta*pi * y))

        model_G = lambda x, y, t: torch.mul(torch.sin(beta * pi * x), torch.sin(beta * pi * y)) + \
                                  zeta*torch.mul(torch.sin(omega*pi*x), torch.sin(omega*pi*y))
        model_D = lambda x, y, t: torch.mul(torch.mul(temp_func(x), temp_func(y)), t / end_time)
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

        u_true = lambda x, y, t: gamma * torch.exp(-alpha * t) * (
                x * (1 - x) * y * (1 - y) + zeta * torch.cos(omega * pi * x) * torch.sin(omega * pi * y))
        u_init = lambda x, y, t: gamma * torch.exp(-alpha * t) * (
                x * (1 - x) * y * (1 - y) + zeta * torch.cos(omega * pi * x) * torch.sin(omega * pi * y))
        u_left = lambda x, y, t: gamma * torch.exp(-alpha * t) * (
                x * (1 - x) * y * (1 - y) + zeta * torch.cos(omega * pi * x) * torch.sin(omega * pi * y))
        u_right = lambda x, y, t: gamma * torch.exp(-alpha * t) * (
                x * (1 - x) * y * (1 - y) + zeta * torch.cos(omega * pi * x) * torch.sin(omega * pi * y))
        u_bottom = lambda x, y, t: gamma * torch.exp(-alpha * t) * (
                x * (1 - x) * y * (1 - y) + zeta * torch.cos(omega * pi * x) * torch.sin(omega * pi * y))
        u_top = lambda x, y, t: gamma * torch.exp(-alpha * t) * (
                x * (1 - x) * y * (1 - y) + zeta * torch.cos(omega * pi * x) * torch.sin(omega * pi * y))
        f_side = lambda x, y, t: get_fside2Multiscale_Dirichlet_2d_2(
            x, y, t, alpha=alpha, gamma=gamma, omega=omega, vx=vx, vy=vy, kx=kx, ky=ky, PI=torch.pi, zeta=zeta)

        # model_G = lambda x, y, t: gamma*torch.mul(x*(1-x), y*(1-y))
        # model_G = lambda x, y, t: torch.mul(x * (1 - x), y * (1 - y))
        # model_G = lambda x, y, t: gamma*torch.sin(pi * x) * torch.sin(pi * y)

        # model_G = lambda x, y, t: gamma * (x*(1 - x)*y*(1 - y) + torch.cos(2.0 * pi * x) * torch.sin(2.0 * pi * y))

        model_G = lambda x, y, t: gamma * (
                        x * (1 - x) * y * (1 - y) + zeta * torch.cos(omega * pi * x) * torch.sin(omega * pi * y))

        model_D = lambda x, y, t: torch.mul(torch.mul(temp_func(x), temp_func(y)), t / end_time)

    mscalednn = HardPINN(input_dim=R['input_dim'], out_dim=R['output_dim'], hidden_layer=R['hidden_layers'],
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
    optimizer = torch.optim.Adam(params2Net, lr=R['learning_rate'])                # Adam

    # 定义更新学习率的方法
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.995)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.975)

    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    test_epoch = []
    loss_init_all = []

    test_bach_size = 100

    if R['testData_model'] == 'random_generate':
        test_bach_size = 4000
        random_coord2x = (region_r - region_l) * np.random.rand(test_bach_size, 1) + region_l
        random_coord2y = (region_t - region_b) * np.random.rand(test_bach_size, 1) + region_t

        test_xy_bach = np.concatenate([random_coord2x, random_coord2y], axis=-1, dtype=np.float32)
        test_time_batch = 0.5 * np.ones(shape=[test_bach_size, 1], dtype=np.float32)
    elif R['testData_model'] == 'load_RegularDomain_Data':
        size2test = 258
        mat_data_path = '../data2RegularDomain_2D'
        test_xy_bach = get_meshData2Advection(data_path=mat_data_path, mesh_number=7, to_torch=False, to_float=True,
                                              to_cuda=False, gpu_no=0, use_grad2x=False)
        shape2xy = np.shape(test_xy_bach)
        batch2test = shape2xy[0]
        test_time_batch = 0.5 * np.ones(shape=[batch2test, 1], dtype=np.float32)
    else:
        if R['equa_name'] == 'Smooth_Dirichlet2':
            test_xy_bach = dataUtilizer2torch.load_data2porous_domain(
                region_left=region_l, region_right=region_r, region_bottom=region_b, region_top=region_t,
                to_torch=False, to_float=True, to_cuda=False, gpu_no=0, use_grad2x=False, scale_trans=True,
                scale2x=region_r - region_l, scale2y=region_t - region_b, base2x=region_l, base2y=region_b)
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

    t_init_batch = np.ones(shape=[batchsize_init, 1], dtype=np.float32) * init_time
    t_init_batch = torch.from_numpy(t_init_batch)
    if True == R['use_gpu']:
        t_init_batch = t_init_batch.cuda(device='cuda:' + str(R['gpuNo']))

    t0 = time.time()
    for i_epoch in range(R['max_epoch'] + 1):
        # 生成训练的时候的内部点
        xy_it_batch = dataUtilizer2torch.rand_in_2D(
            batch_size=batchsize_it, variable_dim=R['input_dim'] - 1, region_left=region_l, region_right=region_r,
            region_bottom=region_b, region_top=region_t, to_torch=True, to_float=True, to_cuda=R['use_gpu'],
            gpu_no=0, use_grad2x=True)

        t_it_batch = dataUtilizer2torch.rand_in_1D(
            batch_size=batchsize_it, variable_dim=1, region_a=init_time, region_b=end_time, to_torch=True,
            to_float=True, to_cuda=R['use_gpu'], gpu_no=0, use_grad2x=True)

        # 计算各个损失来源
        # PDE损失
        UNN2train, loss_it = mscalednn.loss_in(XY=xy_it_batch, t=t_it_batch, loss_type=R['loss_type'], kx=kx,
                                               ky=ky, vx=vx, vy=vy, fside=f_side, model2Dist=model_D,
                                               model2Gene=model_G, if_NN2model_D=False, if_NN2model_G=False)

        # 边界损失点之和
        loss_bd = 0
        # loss = loss_it + loss_bd + loss_init
        loss = loss_it
        Uexact2train = u_true(torch.reshape(xy_it_batch[:, 0], shape=[-1, 1]),
                              torch.reshape(xy_it_batch[:, 1], shape=[-1, 1]),
                              torch.reshape(t_it_batch, shape=[-1, 1]))

        loss_it_all.append(loss_it.item())
        loss_bd_all.append(loss_bd)
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

            test_epoch.append(i_epoch / 1000)

            UNN2test = mscalednn.eval_MscaleDNN(XY_points=test_xy_torch, t_points=test_time_torch, model2Dist=model_D,
                                                model2Gene=model_G, if_NN2model_D=False, if_NN2model_G=False)
            point_square_error = torch.square(Utrue2test - UNN2test)
            test_mse = torch.mean(point_square_error)
            test_rel = test_mse / torch.mean(torch.square(Utrue2test))
            test_mse_all.append(test_mse.item())
            test_rel_all.append(test_rel.item())
            DNN_tools.print_and_log_train_one_epoch_pinn2hard(i_epoch, run_times, tmp_lr, loss_it.item(), loss_bd,
                                                              loss.item(), train_mse.item(), train_rel.item(),
                                                              log_out=log_fileout)
            DNN_tools.print_and_log_test_one_epoch(test_mse.item(), test_rel.item(), log_out=log_fileout)

    # ----------------------  save testing results to mat files, then plot them --------------------------------
    if True == R['use_gpu']:
        utrue2test_numpy = Utrue2test.cpu().detach().numpy()
        unn2test_numpy = UNN2test.cpu().detach().numpy()
    else:
        utrue2test_numpy = Utrue2test.numpy()
        unn2test_numpy = UNN2test.numpy()

    saveData.save_2testSolus2mat(utrue2test_numpy, unn2test_numpy, actName='utrue',
                                 actName1=R['name2act_hidden'], outPath=R['FolderName'])

    # plotData.plot_Hot_solution2test(utrue2test_numpy, size_vec2mat=size2test, actName='Utrue',
    #                                 seedNo=R['seed'], outPath=R['FolderName'])
    # plotData.plot_Hot_solution2test(unn2test_numpy, size_vec2mat=size2test, actName=R['name2act_hidden'],
    #                                 seedNo=R['seed'], outPath=R['FolderName'])

    saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=R['name2act_hidden'], outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=R['name2act_hidden'],
                              outPath=R['FolderName'], yaxis_scale=True)


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
    # R['model2NN'] = 'DNN'
    # R['model2NN'] = 'Scale_DNN'
    R['model2NN'] = 'Fourier_DNN'
    # R['model2NN'] = 'Fourier_SubDNN'
    R['max_epoch'] = 50000
    # 文件保存路径设置
    if R['model2NN'] == 'Fourier_SubDNN':
        store_file = 'HardFourierSub2Dirichlet_2D'
    elif R['model2NN'] == 'Fourier_DNN':
        store_file = 'HardFourier2Dirichlet_2D'

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
    R['input_dim'] = 3  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数

    R['PDE_type'] = 'HardPINN'
    # R['equa_name'] = 'Smooth_Dirichlet'
    R['equa_name'] = 'Smooth_Dirichlet2'
    # R['equa_name'] = 'Multiscale_Dirichlet'
    # R['equa_name'] = 'Multiscale_Dirichlet2'

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置(内部和边界)
    R['batch_size2interior'] = 8000  # 内部训练数据的批大小
    R['batch_size2boundary'] = 1000  # 边界训练数据的批大小
    R['batch_size2init'] = 3000
    R['batch_size2test'] = 100

    # 装载测试数据模式
    R['testData_model'] = 'load_porous_Data'
    # R['testData_model'] = 'load_RegularDomain_Data'
    # R['testData_model'] = 'random_generate'

    R['loss_type'] = 'L2_loss'  # loss类型:L2 loss
    # R['loss_type'] = 'lncosh_loss'
    R['lambda2lncosh'] = 0.5

    R['optimizer_name'] = 'Adam'  # 优化器
    R['learning_rate'] = 0.01  # 学习率
    # R['learning_rate'] = 0.001              # 学习率
    R['learning_rate_decay'] = 5e-5  # 学习率 decay
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
