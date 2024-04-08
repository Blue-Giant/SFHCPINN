from torch import nn
import torch.nn.functional as tnf
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch


class my_actFunc(nn.Module):
    def __init__(self, actName='linear'):
        super(my_actFunc, self).__init__()
        self.actName = actName

    def forward(self, x_input):
        if str.lower(self.actName) == 'relu':
            out_x = tnf.relu(x_input)
        elif str.lower(self.actName) == 'leaky_relu':
            out_x = tnf.leaky_relu(x_input)
        elif str.lower(self.actName) == 'tanh':
            out_x = torch.tanh(x_input)
        elif str.lower(self.actName) == 'enhance_tanh' or str.lower(self.actName) == 'enh_tanh':  # Enhance Tanh
            out_x = torch.tanh(0.5*torch.pi*x_input)
        elif str.lower(self.actName) == 'srelu':
            out_x = tnf.relu(x_input)*tnf.relu(1-x_input)
        elif str.lower(self.actName) == 's2relu':
            out_x = tnf.relu(x_input)*tnf.relu(1-x_input)*torch.sin(2*torch.pi*x_input)
        elif str.lower(self.actName) == 'elu':
            out_x = tnf.elu(x_input)
        elif str.lower(self.actName) == 'sin':
            out_x = torch.sin(x_input)
        elif str.lower(self.actName) == 'sinaddcos':
            out_x = 0.5*torch.sin(x_input) + 0.5*torch.cos(x_input)
            # out_x = 0.75*torch.sin(x_input) + 0.75*torch.cos(x_input)
            # out_x = torch.sin(x_input) + torch.cos(x_input)
        elif str.lower(self.actName) == 'fourier':
            out_x = torch.cat([torch.sin(x_input), torch.cos(x_input)], dim=-1)
        elif str.lower(self.actName) == 'sigmoid':
            out_x = tnf.sigmoid(x_input)
        elif str.lower(self.actName) == 'gelu':
            out_x = tnf.gelu(x_input)
        elif str.lower(self.actName) == 'gcu':
            out_x = x_input*torch.cos(x_input)
        elif str.lower(self.actName) == 'mish':
            out_x = tnf.mish(x_input)
        elif str.lower(self.actName) == 'gauss':
            out_x = torch.exp(-1.0 * x_input * x_input)
            # out_x = torch.exp(-0.5 * x_input * x_input)
        elif str.lower(self.actName) == 'requ':
            out_x = tnf.relu(x_input)*tnf.relu(x_input)
        elif str.lower(self.actName) == 'recu':
            out_x = tnf.relu(x_input)*tnf.relu(x_input)*tnf.relu(x_input)
        elif str.lower(self.actName) == 'morlet':
            out_x = torch.cos(1.75*x_input)*torch.exp(-0.5*x_input*x_input)
            # out_x = torch.cos(1.75 * x_input) * torch.exp(-1.0 * x_input * x_input)
        else:
            out_x = x_input
        return out_x


class FCN(nn.Module):
    def __init__(self, indim=2, outdim=1, width=None, actName2In='tanh', actName2Hidden='Tanh', actName2Out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0):
        super(FCN, self).__init__()
        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        # Inputs to hidden layer linear transformation
        self.input_layer = nn.Linear(in_features=indim, out_features=width,
                                     dtype=self.float_type, device=self.opt2device)
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.uniform_(self.input_layer.bias, -1, 1)

        self.hidden_layer1 = nn.Linear(in_features=width, out_features=width,
                                       dtype=self.float_type, device=self.opt2device)
        nn.init.xavier_normal_(self.hidden_layer1.weight)
        nn.init.uniform_(self.hidden_layer1.bias, -1, 1)

        self.hidden_layer2 = nn.Linear(in_features=width, out_features=width,
                                       dtype=self.float_type, device=self.opt2device)
        nn.init.xavier_normal_(self.hidden_layer2.weight)
        nn.init.uniform_(self.hidden_layer2.bias, -1, 1)

        # Hidden layer to output layer, width units --> one for each digit
        self.output_layer = nn.Linear(in_features=width, out_features=outdim,
                                      dtype=self.float_type, device=self.opt2device)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.uniform_(self.output_layer.bias, -1, 1)

        # Define sigmoid activation and softmax output
        self.actFunc_in = my_actFunc(actName=actName2In)
        self.actFunc = my_actFunc(actName=actName2Hidden)
        self.actFunc_out = my_actFunc(actName=actName2Out)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        H_IN = self.input_layer(x)
        H = self.actFunc_in(H_IN)

        H = self.hidden_layer1(H)
        H = self.actFunc(H)

        H = self.hidden_layer2(H)
        H = self.actFunc(H)

        H = self.output_layer(H)
        H_OUT = self.actFunc_out(H)

        return H_OUT


class Fourier_FCN(nn.Module):
    def __init__(self, indim=2, outdim=1, width=None, actName2In='fourier', actName2Hidden='Tanh', actName2Out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0):
        super(Fourier_FCN, self).__init__()
        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        # Inputs to hidden layer linear transformation
        self.input_layer = nn.Linear(in_features=indim, out_features=width,
                                     dtype=self.float_type, device=self.opt2device)
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.uniform_(self.input_layer.bias, -1, 1)

        self.hidden_layer1 = nn.Linear(in_features=2*width, out_features=width,
                                      dtype=self.float_type, device=self.opt2device)
        nn.init.xavier_normal_(self.hidden_layer1.weight)
        nn.init.uniform_(self.hidden_layer1.bias, -1, 1)

        self.hidden_layer2 = nn.Linear(in_features=width, out_features=width,
                                       dtype=self.float_type, device=self.opt2device)
        nn.init.xavier_normal_(self.hidden_layer2.weight)
        nn.init.uniform_(self.hidden_layer2.bias, -1, 1)

        # Hidden layer to output layer, width units --> one for each digit
        self.output_layer = nn.Linear(in_features=width, out_features=outdim,
                                      dtype=self.float_type, device=self.opt2device)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.uniform_(self.output_layer.bias, -1, 1)

        # Define sigmoid activation and softmax output
        self.actFunc_in = my_actFunc(actName=actName2In)
        self.actFunc = my_actFunc(actName=actName2Hidden)
        self.actFunc_out = my_actFunc(actName=actName2Out)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        H_IN = self.input_layer(x)
        H = torch.cat([torch.cos(H_IN), torch.sin(H_IN)], dim=-1)

        H = self.hidden_layer1(H)
        H = self.actFunc(H)

        H = self.hidden_layer2(H)
        H = self.actFunc(H)

        H = self.output_layer(H)
        H_OUT = self.actFunc_out(H)

        return H_OUT


class MultiScale_Fourier_FCN(nn.Module):
    def __init__(self, indim=2, outdim=1, width=None, actName2In='fourier', actName2Hidden='Tanh', actName2Out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, freq=None, repeat_high_freq=False):
        super(MultiScale_Fourier_FCN, self).__init__()
        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.repeat_high_freq = repeat_high_freq

        Unit_num = int(width / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_high_freq == True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([width - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([width - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(gpu_no))

        # Inputs to hidden layer linear transformation
        self.input_layer = nn.Linear(in_features=indim, out_features=width,
                                     dtype=self.float_type, device=self.opt2device)
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.uniform_(self.input_layer.bias, -1, 1)

        self.hidden_layer1 = nn.Linear(in_features=2*width, out_features=width,
                                      dtype=self.float_type, device=self.opt2device)
        nn.init.xavier_normal_(self.hidden_layer1.weight)
        nn.init.uniform_(self.hidden_layer1.bias, -1, 1)

        self.hidden_layer2 = nn.Linear(in_features=width, out_features=width,
                                       dtype=self.float_type, device=self.opt2device)
        nn.init.xavier_normal_(self.hidden_layer2.weight)
        nn.init.uniform_(self.hidden_layer2.bias, -1, 1)

        # Hidden layer to output layer, width units --> one for each digit
        self.output_layer = nn.Linear(in_features=width, out_features=outdim,
                                      dtype=self.float_type, device=self.opt2device)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.uniform_(self.output_layer.bias, -1, 1)

        # Define sigmoid activation and softmax output
        self.actFunc_in = my_actFunc(actName=actName2In)
        self.actFunc = my_actFunc(actName=actName2Hidden)
        self.actFunc_out = my_actFunc(actName=actName2Out)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        H_IN = self.input_layer(x)
        H = torch.cat([torch.cos(H_IN * self.torch_mixcoe), torch.sin(H_IN * self.torch_mixcoe)], dim=-1)

        H = self.hidden_layer1(H)
        H = self.actFunc(H)

        H = self.hidden_layer2(H)
        H = self.actFunc(H)

        H = self.output_layer(H)
        H_OUT = self.actFunc_out(H)

        return H_OUT


class Network_PINN(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.input = nn.Linear(2, 20)
        self.hidden = nn.Linear(20, 20)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(20, 1)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.input(x)
        x = self.sigmoid(x)
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        return x

    def train_process(self, x_train, y_train, num_epoches):
        for epoch in range(num_epoches):
            correct = 0
            total = 0
            run_loss = 0.0
            pred = self(x_train)
            loss = self.criterion(pred, y_train)
            loss.backward()
            self.optimizer.step()
            run_loss = loss.item()
            num = 100
            if epoch % num == num - 1:
                print('epochs:%d   MSEloss : %.3f' % (epoch + 1, run_loss / num))
                run_loss = 0

    def loss_it(self, model_G, model_distance, inputs, loss_type='l2_loss', ws=0.01, ds=0.002):
        x_train, y_train, DXY, dD_dx, dD_dxx, dD_dy, GXY, dG_dx, dG_dxx, dG_dy\
            =inputs(inputs, model_G, model_distance)
        UNN = self(inputs)
        U = GXY + torch.mul(DXY, UNN)
        dU_dx = torch.autograd.grad(UNN, x_train, grad_outputs=torch.ones_like(x_train),
                                    create_graph=True, retain_graph=True, allow_unused=True)[0]
        dU_dy = torch.autograd.grad(UNN, y_train, grad_outputs=torch.ones_like(y_train),
                                    create_graph=True, retain_graph=True, allow_unused=True)[0]
        dU_dxx = torch.autograd.grad(dU_dx, x_train, grad_outputs=torch.ones_like(x_train),
                                     create_graph=True, retain_graph=True, allow_unused=True)[0]

        LG = dG_dy + ws * dG_dx - ds * dG_dxx
        LDU = torch.mul(dD_dy, UNN) + torch.mul(DXY, dU_dy) + ws * (torch.mul(dD_dx, UNN) + torch.mul(DXY, dU_dx))
        LDU2 = ds * (torch.mul(dD_dxx, UNN) + torch.mul(dD_dx, dU_dx) + torch.mul(DXY, dU_dxx))
        LU = LG + LDU - LDU2
        if loss_type =='l2_loss':
            loss_it = torch.square(LU)
        return U, loss_it


def tensorv(x):
    x = torch.FloatTensor(x)
    #     return Variable(x,requires_grad=True)
    return Variable(x, requires_grad=True)


def inputs(inside_points, model_G, model_distance ):
    GXY = model_G(inside_points)
    x_train = inside_points[:, 0]
    y_train = inside_points[:, 1]
    dG_dx = torch.autograd.grad(GXY, x_train, grad_outputs=torch.ones_like(x_train),
                                create_graph=True, retain_graph=True, allow_unused=True)
    dG_dx = dG_dx[0]
    dG_dy = torch.autograd.grad(GXY, y_train, grad_outputs=torch.ones_like(y_train),
                                create_graph=True, retain_graph=True, allow_unused=True)[0]
    dG_dxx = torch.autograd.grad(dG_dx, x_train, grad_outputs=torch.ones_like(x_train),
                                 create_graph=True, retain_graph=True, allow_unused=True)[0]
    DXY = model_distance(inside_points)
    dD_dx = torch.autograd.grad(DXY, x_train, grad_outputs=torch.ones_like(x_train),
                                create_graph=True, retain_graph=True, allow_unused=True)[0]
    dD_dy = torch.autograd.grad(DXY, y_train, grad_outputs=torch.ones_like(y_train),
                                create_graph=True, retain_graph=True, allow_unused=True)[0]
    dD_dxx = torch.autograd.grad(dD_dx, x_train, grad_outputs=torch.ones_like(x_train),
                                 create_graph=True, retain_graph=True, allow_unused=True)[0]
    return x_train, y_train, DXY, dD_dx,dD_dxx, dD_dy, GXY, dG_dx,dG_dxx, dG_dy
