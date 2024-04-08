# 目的
该文件夹下的代码为使用 Transformer 进行算子学习。
将Tranformer与之前的ScaleDNN和FourierDNN进行结合，提升Transformer的性能。

# Solve the advection diffusion equation in one- to three-dimensional space

## One dimensional cases 
### 1、Dirichlet boundary
    (1) solve the genral smooth function u(x,t) = exp(-alpha*t)*sin(2.0*pi*x) with 
        alpha = 0.25, the intrested space domain is [0,2] and the time interval is [0,5]

    (2) solve the multiscale function u(x,t) = exp(-alpha*t)*[sin(2.0*pi*x)+zeta*sin(beta*pi*x)]
        with alpha=0.25, beta=50, zeta=0.1, the intrested space domain is [0,1] and the time interval is [0,1]

    (3) In terms of hard version PINN, the distance function can be set as analytical or NN. For general case, 
        analytical function model_D = x(2-x)/[(2-0)*(2-0)]*(t/5)
        analytical function model_G = sin(2.0*pi*x)

        the model_D represented by DNN should satisfy the distance between inerior and boundary tends to zero.
        the model_G represented by DNN should satisfy the boundary conditions

    (4) For multiscale case:
        analytical function model_D = x(1-x)/[(1-0)*(1-0)]*(t/1)
        analytical function model_G_1 = sin(2.0*pi*x) or model_G_2 = sin(2.0*pi*x)+zeta*sin(beta*pi*x), in which 
        the performasnce of mode_G_2 is better than that of mode_G_1

        the model_D represented by DNN should satisfy the distance between inerior and boundary tends to zero.
        the model_G represented by DNN should satisfy the boundary conditions


### 2、Neumann boundary
    (1) solve the genral smooth function u(x,t) = exp(-alpha*t)*sin(pi*x) with 
        alpha = 0.25, the intrested space domain is [0,2] and the time interval is [0,5]

    (2) solve the multiscale function u(x,t) = exp(-alpha*t)*[sin(pi*x)+zeta*sin(omega*pi*x)]
        with alpha=0.25, omega=20, zeta=0.2, the intrested space domain is [0,1] and the time interval is [0,1]

    (3) For general case, 
        analytical function model_D1 = x(2-x)/[(2-0)*(2-0)]*(t/5) or model_D1 = 1-exp(-t)
        analytical function model_G = sin(pi*x)

        the model_D represented by DNN should satisfy the distance between inerior and boundary tends to zero.
        the model_G represented by DNN should satisfy the boundary conditions

## two dimensional cases 
### 1、Dirichlet boundary
    solve problem in porous domain in 2D, refer to: data_path = '../data2PorousDomain_2D/Normalized/xy_porous5.txt'
    (1)


### 2、Neumann boundary
    solve problem in regular square domain in 2D 

## Tips
### 1、The choice of extension function is more important than that of distance function for Dirichlet problem