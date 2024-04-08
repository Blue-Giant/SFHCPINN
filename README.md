# Title

# Abstrct
Deep learning methods have gained considerable interest in the numerical solution of various partial differential equations (PDEs). One particular focus is physics-informed neural networks (PINN), which integrate physical principles into neural networks. This transforms the process of solving PDEs into optimization problems for neural networks. To address a collection of advection-diffusion equations (ADE) in a range of difficult circumstances, this paper proposes a novel network structure. This architecture integrates the solver, a multi-scale deep neural networks (MscaleDNN) utilized in the PINN method, with a hard constraint technique known as HCPINN. This method introduces a revised formulation of the desired solution for ADE by utilizing a loss function that incorporates the residuals of the governing equation and penalizes any deviations from the specified boundary and initial constraints. By surpassing the boundary constraints automatically, this method improves the accuracy and efficiency of the PINN technique. To address the “spectral bias” phenomenon in neural networks, a subnetwork structure of MscaleDNN and a Fourier-induced activation function are incorporated into the HCPINN, resulting in a hybrid approach called SFHCPINN. The effectiveness of SFHCPINN is demonstrated through various numerical experiments involving ADE in different dimensions. The numerical results indicate that SFHCPINN outperforms both standard PINN and its subnetwork version with Fourier feature embedding. It achieves remarkable accuracy and efficiency while effectively handling complex boundary conditions and high-frequency scenarios in ADE.

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
