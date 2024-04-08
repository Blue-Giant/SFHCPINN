# Idea for SFHCPINN
Solving a class of elliptic partial differential equations(PDEs) with multiple scales utilizing Fourier-based mixed physics informed neural networks(dubbed FMPINN), the solver of FMPINN is configured as a multi-scale deep neural networks.

# Title of paper
Physical informed neural networks with soft and hard boundary constraints for solving advection-diffusion equations using Fourier expansions

[[Paper]](https://pdf.sciencedirectassets.com/271503/1-s2.0-S0898122124X00048/1-s2.0-S0898122124000348/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIBAsWAIcTZyP7zZDYn2NlImbiBMnb%2BstbRPZ7Ka%2BuRUQAiEAuRUVZXtNOewDD4bqmSwbf%2BRFs6dv9ox2RsE0Nuv2hJIqvAUI7P%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDC7wnMP%2FI4I5bbcayyqQBcXZ5h%2BthpDdYTFc4Rz7QMnf2lDXHW6UTEnd26EIKklHYXkOhyfTZIBKI2rW9Hxeswy8BCi%2BRIa2oTSGa%2BCYXuBpt4k4V5ebT020zkjQ0lFpJIzcUUpz3PPOJaqB6p3sgf4I3ssEa0dyTRgD0HJWUqYQvKhJ%2FinIYLbKME2ZE6W4E%2B9H8JmCpInQBVtJ%2BA%2BCmxVZtHQOq53%2BnT7%2FCdk9jApRGMNGtOzNTLc2nv37pxITTX9kCyVKERBgK%2FbSRSFrd50uywIIExmvjlnkIlzIEMq%2Biy%2FVQ%2Bm3msoCkBslQmVDQ35fXiBO7%2Fp3AAdd9WN%2BiO%2FSI%2FAFPxgW%2F2MrMLHMV0KpjMWwqg5M%2BrY%2BvZGMrPbd0B52bi9%2FLn%2Bzaf5Gj1Vty%2Bjd7QaUzlUW27gB6Ymuji580c%2FBm%2BAeaEbTm1JuQ1jcuH7SIscef4ctPWRc%2Bf6lJRlnbVbXRxiw72tk8MJvkM31zFFDTxmV1TInvFY2ueK%2FiqdoUtsXrlTHp%2Fya%2Be9CcQKH89HdRuXIsGKl8aT9SHcILctNNJLMy%2B%2B1yuz5pCA5SIFvvqF3YJ87qr81Ka%2F6eefx2xtBByLc47RaOcinWUAJuW%2FfFDrAOLjTBbng%2BkhiF0lQd%2FwYp2S4WxyEU3%2FAw%2FsH97N1bXZ17i8t%2Fy47R5333hh1DZE6ZfNf3OU0q0ghRisI0se4MQCK8ItULSfbsM8nK8bWj8F91v%2Boq%2BgZ4i%2BOvhVtz5MOCtXYZTgpEvcdiQEgtIp0RjOWe4NH01J98gyv5mK4MkYO5w9uGzOtoy1Y1hGzyH3%2BIZwuy%2FzylCXG5j6L%2FpWBNIkFakp09oghCKLXXUSlKhY3nrzXzCRVWvAubzC7b8OR7P7CRGq2dDe7MKaTz7AGOrEBCwMCVK5cj%2FQGrhPi1TezD0aje8X5jnJFO%2FfluZO8uaW22Cw%2B1hcHu6%2FGIvzH%2Bg6rwuAZv0Hu5IhfxI5rwNPIQgIM6giK5uO%2B5m3DXJ0ZrU8xDGiofbr%2B%2FouINNRxwP7pD1sAnfAwniVc6SqhVHru3Kujba4LDZdwiLiKtE3Rf4iCf5UHgkJ%2FN%2Fvw6wkZn5ah4cuHV18GFg5lLBfM5grgR63oJyg4ix9QihrfPYhSRjrS&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240408T120736Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY5QIZYVAX%2F20240408%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=2f8089f10231535d93f4e3a212c1152e0f16053a2616252c78d6c80ebfe0a42f&hash=75e8c5f2b96231747f47a066579e8bea78edc6f2df3620e71dd6eb913bce11df&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0898122124000348&tid=spdf-cb3afe24-1795-442b-98c7-fd65b568bf11&sid=304b269c136bd74f2c8a7a1-30f930048898gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=06095c56555c545d5250&rr=871223534bb36423&cc=cn)

# Abstract
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
