import os
import sys
import torch
import numpy as np
from Networks.NETWORK import *
from Utilizers.gen_data import *
import matplotlib.pyplot as plt


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


def get_fside2multi_scale(x, t, alpha=0.25, zeta=0.1, omega=10, ws=0.1, ds=0.1, PI=torch.pi):
    exp_t = torch.exp(-alpha * t)
    sin_PIx_sinOmegax = torch.sin(PI * x) + zeta * torch.sin(omega * PI * x)
    ut = -alpha*torch.mul(exp_t, sin_PIx_sinOmegax)
    ux = torch.mul(exp_t, PI*torch.cos(PI*x) + zeta*omega*PI*torch.cos(omega*PI*x))
    uxx = torch.mul(exp_t, -1.0*PI*PI*torch.sin(PI*x)-1.0*zeta*omega*omega*PI*PI*torch.sin(omega*PI*x))
    f = ut - ds*uxx + ws*ux
    return f


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
        u_init = lambda x, t: torch.sin(torch.pi * x)*torch.ones_like(t)
        f_side = lambda x, t: get_fside2general(x, t, alpha=alpha, beta=1.0, ds=ds, ws=ws)
    elif R['equa_name'] == '1DV_MultiScale':
        region_l = 0.0
        region_r = 1.0
        init_time = 0.0
        end_time = 1.0
        beta = 1
        alpha = 0.25  # 二次项
        omega = 20
        zeta = 0.2
        ws = 1.0
        ds = 0.1
        u_true = lambda x, t: torch.mul(torch.exp(-alpha*t), torch.sin(torch.pi*x)+zeta*torch.sin(omega*torch.pi*x))
        u_left = lambda x, t: torch.exp(-alpha * t) * (torch.pi*torch.cos(torch.pi*x)+zeta*omega*torch.pi*torch.cos(omega*torch.pi*x))
        u_right = lambda x, t: torch.exp(-alpha * t) * (torch.pi*torch.cos(torch.pi*x)+zeta*omega*torch.pi*torch.cos(omega*torch.pi*x))
        u_init = lambda x, t: torch.sin(torch.pi*x)+zeta*torch.sin(omega*torch.pi*x)
        f_side = lambda x, t: get_fside2multi_scale(x, t, alpha=alpha, zeta=zeta, omega=omega, ws=ws, ds=ds)

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
            to_float=True, to_torch=True, to_cuda=False, gpu_no=0, use_grad=True)

        if True == Rdic['with_gpu']:
            xt_left = xt_left.cuda(device='cuda:' + str(Rdic['no2gpu']))
            xt_right = xt_right.cuda(device='cuda:' + str(Rdic['no2gpu']))
            xt_init = xt_init.cuda(device='cuda:' + str(Rdic['no2gpu']))

        gnn_left = model_extension(xt_left)
        gnn_right = model_extension(xt_right)
        gnn_init = model_extension(xt_init)

        grad2gnn_left = torch.autograd.grad(gnn_left, xt_left, grad_outputs=torch.ones_like(gnn_left),
                                            create_graph=True, retain_graph=True)[0]
        grad2dgnn_right = torch.autograd.grad(gnn_right, xt_right, grad_outputs=torch.ones_like(gnn_right),
                                              create_graph=True, retain_graph=True)[0]

        dgnn_left = torch.reshape(grad2gnn_left[:, 0], shape=[-1, 1])
        dgnn_right = torch.reshape(grad2dgnn_right[:, 0], shape=[-1, 1])

        g_left = u_left(torch.reshape(xt_left[:, 0], shape=[-1, 1]), torch.reshape(xt_left[:, 1], shape=[-1, 1]))
        g_right = u_right(torch.reshape(xt_right[:, 0], shape=[-1, 1]), torch.reshape(xt_right[:, 1], shape=[-1, 1]))
        g_init = u_init(torch.reshape(xt_init[:, 0], shape=[-1, 1]), torch.reshape(xt_init[:, 1], shape=[-1, 1]))

        diff_left = dgnn_left - g_left
        diff_right = dgnn_right - g_right
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

    # R['equa_name'] = '1DV_General'
    R['equa_name'] = '1DV_MultiScale'                 # Example in paper

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
    R['scale_list'] = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 14])
    R['repeat_high_freq'] = False

    modelG = general_extension_func(Rdic=R)

    if R['equa_name'] == '1DV_General':
        torch.save(modelG, "Neumann1D_ExtensionModel2Space02Time05.pth")  # 保存整个模型
    elif R['equa_name'] == '1DV_MultiScale':
        torch.save(modelG, "Neumann1D_ExtensionModel2Space01Time01.pth")  # 保存整个模型
