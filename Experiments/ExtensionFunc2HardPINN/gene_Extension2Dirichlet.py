import os
import sys
import torch
import numpy as np
from Networks.NETWORK import *
from Utilizers.gen_data import *
import matplotlib.pyplot as plt


def f_side2smooth(x, t, ws=0.001, ds=0.002, alpha=0.001, beta=20, zeta=1.0, PI=torch.pi):
    exp_t = torch.exp(-alpha*t)
    pu_pt = -alpha*exp_t*torch.sin(2*PI*x)
    pu_px = exp_t*(2*PI*torch.cos(2*PI*x))
    ppu_pxx = exp_t*(-4*PI*PI*torch.sin(2*PI*x))
    f_side = pu_pt - ds*ppu_pxx + ws*pu_px
    return f_side


def f_side_case1(x, t, ws=0.001, ds=0.002, alpha=0.001, beta=20, zeta=0.01, PI=torch.pi):
    exp_t = torch.exp(-alpha * t)
    sin_2pi_beta_pi = torch.sin(2 * PI * x) + zeta * torch.sin(beta * PI * x)
    pu_pt = -alpha * exp_t * sin_2pi_beta_pi

    item2pu_px = 2 * PI * torch.cos(2 * PI * x) + zeta * beta * PI * torch.cos(beta * PI * x)
    pu_px = exp_t * item2pu_px

    item2ppu_pxx = -4 * PI * PI * torch.sin(2 * PI * x) - zeta * beta * PI * beta * PI * torch.sin(beta * PI * x)
    ppu_pxx = exp_t * item2ppu_pxx
    f_side = pu_pt - ds * ppu_pxx + ws * pu_px
    return f_side


def f_side_case_2pi_50pi(x, t, ws=0.001, ds=0.002, alpha=0.001, beta=20, zeta=0.01, PI=torch.pi):
    exp_t = torch.exp(-alpha*t)
    sin_2pi_beta_pi = torch.sin(2 * PI * x) + zeta*torch.sin(beta * PI * x)
    pu_pt = -alpha*exp_t*sin_2pi_beta_pi

    item2pu_px = 2 * PI * torch.cos(2*PI*x) + zeta * beta * PI * torch.cos(beta * PI * x)
    pu_px = exp_t*item2pu_px

    item2ppu_pxx = -4 * PI * PI * torch.sin(2 * PI * x) - zeta * beta * PI * beta * PI * torch.sin(beta * PI * x)
    ppu_pxx = exp_t * item2ppu_pxx
    f_side = pu_pt - ds*ppu_pxx + ws*pu_px
    return f_side


def gene_init_bd_points2double_dirichlet(point_num2bd=1000, point_num2init=100, region_l=0.0, region_r=1.0,
                                         time_begin=0.1, time_end=1.0, variable_dim=2, opt2sampler='lhs', to_float=True,
                                         to_torch=True, to_cuda=False, gpu_no=0, use_grad=False):
    # # 迪利克雷边界: 随机生成区域边界点（空间左右边界，时间初始边界）---------- 初始条件可以看作迪利克雷边界
    assert (int(variable_dim) == 2)
    if str.lower(opt2sampler) == 'random':
        t2left_right = (time_end - time_begin) * np.random.random([point_num2bd, 1]) + time_begin

        x2left = np.ones([point_num2bd, 1]) * region_l
        xt_left = np.concatenate([x2left, t2left_right], axis=-1)

        x2right = np.ones([point_num2bd, 1]) * region_r
        xt_right = np.concatenate([x2right, t2left_right], axis=-1)

        x2begin_end = (region_r - region_l) * np.random.random([point_num2init, 1]) + region_l
        t2begin = np.ones([point_num2init, 1]) * time_begin
        xt_begin = np.concatenate([x2begin_end, t2begin], axis=-1)
    else:
        sampler2time = stqmc.LatinHypercube(d=1)
        sampler2x = stqmc.LatinHypercube(d=1)
        t2left_right = (time_end - time_begin) * sampler2time.random(point_num2bd) + time_begin

        x2left = np.ones([point_num2bd, 1]) * region_l
        xt_left = np.concatenate([x2left, t2left_right], axis=-1)

        x2right = np.ones([point_num2bd, 1]) * region_r
        xt_right = np.concatenate([x2right, t2left_right], axis=-1)

        x2begin_end = (region_r - region_l) * sampler2x.random(point_num2init) + region_l
        t2begin = np.ones([point_num2init, 1]) * time_begin
        xt_begin = np.concatenate([x2begin_end, t2begin], axis=-1)

    if to_float:
        xt_left = xt_left.astype(np.float32)
        xt_right = xt_right.astype(np.float32)
        xt_begin = xt_begin.astype(np.float32)

    if to_torch:
        xt_left = torch.from_numpy(xt_left)
        xt_right = torch.from_numpy(xt_right)
        xt_begin = torch.from_numpy(xt_begin)
        if to_cuda:
            xt_left = xt_left(device='cuda:' + str(gpu_no))
            xt_right = xt_right(device='cuda:' + str(gpu_no))
            xt_begin = xt_begin(device='cuda:' + str(gpu_no))

        xt_left.requires_grad = use_grad
        xt_right.requires_grad = use_grad
        xt_begin.requires_grad = use_grad

    return xt_left, xt_right, xt_begin


def general_extension_func(Rdic=None):
    if Rdic['equa_name'] == 'General_ADE':  # 双诺依曼边界
        # equation EX1
        # The numerical solution of advection–diffusion problems using new cubic trigonometric B-splines approach
        region_l = 0.0
        region_r = 2.0
        init_time = 0.0
        end_time = 5
        ds = 1/ 128
        ws = 1
        u_true = lambda x, t: torch.mul(torch.exp(-ds * t), torch.sin(x - ws * t))
        u_left = lambda x, t: torch.mul(torch.exp(-ds * t), torch.sin(-ws * t))
        u_right = lambda x, t: torch.mul(torch.exp(-ds * t), torch.sin(region_r - ws * t))
        u_init = lambda X, t: torch.sin(X)
    elif Rdic['equa_name'] == 'ADE1d_Sin(2pi)':
        region_l = 0.0
        region_r = 2.0
        init_time = 0.0
        end_time = 5
        ws = 0.001
        ds = 0.002
        pi = np.pi
        alpha = 0.25
        beta = 50
        zeta = 0.2
        f_side = lambda x, t: f_side2smooth(x=x, t=t, ws=ws, ds=ds, alpha=alpha, beta=beta, zeta=zeta)
        u_init = lambda x, t: torch.sin(2 * pi * x) * torch.ones_like(t)
        u_left = lambda x, t: torch.zeros_like(x)
        u_right = lambda x, t: torch.zeros_like(x)
        u_true = lambda x, t: torch.mul(torch.exp(-alpha * t), torch.sin(2 * pi * x))
    elif Rdic['equa_name'] == 'ADE1d_Sin(pi)+0.1sin(30Pi)':
        region_l = 0.0
        region_r = 2.0
        init_time = 0.0
        end_time = 5.0
        ws = 0.01
        ds = 0.02
        pi = np.pi
        alpha = 0.25
        beta = 30
        zeta = 0.1
        f_side = lambda x, t: f_side_case1(x=x, t=t, ws=ws, ds=ds, alpha=alpha, beta=beta, zeta=zeta)
        u_init = lambda x, t: torch.sin(pi * x) + zeta * torch.sin(beta * pi * x)
        u_left = lambda x, t: torch.mul(torch.exp(-alpha * t), torch.sin(pi * x) + zeta * torch.sin(beta * pi * x))
        u_right = lambda x, t: torch.mul(torch.exp(-alpha * t), torch.sin(pi * x) + zeta * torch.sin(beta * pi * x))
        u_true = lambda x, t: torch.mul(torch.exp(-alpha * t), u_init(x, t))
    elif Rdic['equa_name'] == 'ADE1d_Sin(2pi)+0.1sin(50Pi)':
        region_l = 0.0
        region_r = 1.0
        init_time = 0.0
        end_time = 1
        ws = 0.001
        ds = 0.002
        pi = np.pi
        alpha = 0.25
        beta = 50
        zeta = 0.1
        f_side = lambda x, t: f_side_case_2pi_50pi(x=x, t=t, ws=ws, ds=ds, alpha=alpha, beta=beta, zeta=zeta)
        u_init = lambda x, t: torch.sin(2 * pi * x) + zeta * torch.sin(beta * pi * x)
        u_left = lambda x, t: torch.zeros_like(x)
        u_right = lambda x, t: torch.zeros_like(x)
        u_true = lambda x, t: torch.mul(torch.exp(-alpha * t), torch.sin(2 * pi * x) + zeta * torch.sin(beta * pi * x))

    if Rdic['name2model'] == 'Fourier_DNN':
        model_extension = Fourier_FCN(indim=2, outdim=1, width=Rdic['width2nn'], actName2In=Rdic['actIn_Name'],
                                      actName2Hidden=Rdic['actHidden_Name'], actName2Out=Rdic['actOut_Name'],
                                      type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=Rdic['no2gpu'])
    elif Rdic['name2model'] == 'DNN':
        model_extension = FCN(indim=2, outdim=1, width=Rdic['width2nn'], actName2In=Rdic['actIn_Name'],
                              actName2Hidden=Rdic['actHidden_Name'], actName2Out=Rdic['actOut_Name'],
                              type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=Rdic['no2gpu'])
    else:
        model_extension = MultiScale_Fourier_FCN(
            indim=2, outdim=1, width=Rdic['width2nn'], actName2In=Rdic['actIn_Name'],
            actName2Hidden=Rdic['actHidden_Name'], actName2Out=Rdic['actOut_Name'],
            type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=Rdic['no2gpu'], freq=Rdic['scale_list'],
            repeat_high_freq=R['repeat_high_freq'])

    params2Net = model_extension.parameters()
    optimizer = torch.optim.Adam(params2Net, lr=Rdic['init_learning_rate'])  # Adam
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=Rdic['gamma2stepLR'])
    print('---------------------- start training Model of Extension --------------------------')

    for epoch in range(Rdic['max_epochs']+1):
        xt_left, xt_right, xt_init = gene_init_bd_points2double_dirichlet(
            point_num2bd=Rdic['num2in_point'], point_num2init=Rdic['num_inflowbd_points'], region_l=region_l,
            region_r=region_r, time_begin=init_time, time_end=end_time, variable_dim=2, opt2sampler='lhs',
            to_float=True, to_torch=True, to_cuda=False, gpu_no=0, use_grad=False)

        if True == Rdic['with_gpu']:
            xt_left = xt_left.cuda(device='cuda:' + str(Rdic['no2gpu']))
            xt_right = xt_right.cuda(device='cuda:' + str(Rdic['no2gpu']))
            xt_init = xt_init.cuda(device='cuda:' + str(Rdic['no2gpu']))

        gnn_left = model_extension(xt_left)
        gnn_right = model_extension(xt_right)
        gnn_init = model_extension(xt_init)

        g_left = u_left(torch.reshape(xt_left[:, 0], shape=[-1, 1]), torch.reshape(xt_left[:, 1], shape=[-1, 1]))
        g_right = u_right(torch.reshape(xt_right[:, 0], shape=[-1, 1]), torch.reshape(xt_right[:, 1], shape=[-1, 1]))
        g_init = u_init(torch.reshape(xt_init[:, 0], shape=[-1, 1]), torch.reshape(xt_init[:, 1], shape=[-1, 1]))

        diff_left = gnn_left - g_left
        diff_right = gnn_right - g_right
        diff_init = gnn_init - g_init
        loss = torch.mean(torch.square(diff_left)) + torch.mean(torch.square(diff_right)) + \
               torch.mean(torch.square(diff_init))

        optimizer.zero_grad()  # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()        # 对loss关于Ws和Bs求偏导
        optimizer.step()       # 更新参数Ws和Bs
        scheduler.step()

        if epoch % 100 == 0:
            tmp_lr = optimizer.param_groups[0]['lr']
            print('epochs:%d  ----lr:%.8f  -----mse: %.15f' % (epoch, tmp_lr, loss.item()))
    return model_extension


if __name__ == "__main__":
    R={}

    # R['equa_name'] = 'General_ADE'
    # R['equa_name'] = 'ADE1d_Sin(2pi)'                 # Example in paper

    R['equa_name'] = 'ADE1d_Sin(pi)+0.1sin(30Pi)'   # Example in paper

    # R['equa_name'] = 'ADE1d_Sin(2pi)+0.1sin(50Pi)'

    # R['name2model'] = 'DNN'
    # R['name2model'] = 'Fourier_DNN'
    R['name2model'] = 'MultiScale_Fourier'

    R['init_learning_rate'] = 0.01
    R['gamma2stepLR'] = 0.95
    R['max_epochs'] = 20000

    R['num2in_point'] = 2500
    R['num_inflowbd_points'] = 250

    R['width2nn'] = 50
    # R['actIn_Name'] = 'sin'
    # R['actHidden_Name'] = 'sin'

    R['actIn_Name'] = 'tanh'
    R['actHidden_Name'] = 'tanh'
    R['actOut_Name'] = 'linear'
    R['with_gpu'] = True

    R['no2gpu'] = 0
    R['scale_list'] = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20])
    R['repeat_high_freq'] = False

    modelG = general_extension_func(Rdic=R)

    if R['equa_name'] == 'ADE1d_Sin(2pi)' or R['equa_name'] == 'ADE1d_Sin(pi)+0.1sin(30Pi)':
        torch.save(modelG, "Dirichlet1D_ExtensionModel2Space02Time05.pth")  # 保存整个模型
    elif R['equa_name'] == 'ADE1d_Sin(2pi)+0.1sin(50Pi)':
        torch.save(modelG, "Dirichlet1D_ExtensionModel2Space01Time01.pth")  # 保存整个模型
