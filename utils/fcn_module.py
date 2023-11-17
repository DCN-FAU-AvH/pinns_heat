import torch
import torch.autograd as autograd  # computation graph
import torch.nn as nn  # neural networks
import numpy as np
import deepxde as dde

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Neural Network
class FCN(nn.Module):
    ## Neural Network
    def __init__(self, layers, case, domain):
        super().__init__()  # call __init__ from parent class
        "activation function"
        self.layers = layers
        self.case = case
        self.domain = domain
        self.activation = nn.Tanh()  # For more activation functions see: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        "loss function"
        self.loss_function = nn.MSELoss(reduction="mean")
        "Initialise neural network as a list using nn.Modulelist"
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)])
        self.iter = 0  # For the Optimizer
        "Xavier Normal Initialization"
        for i in range(len(self.layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)

    "Foward pass"

    def forward(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(self.layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

    "Loss Functions"

    # Loss BC
    def lossBC(self, x_BC, y_BC):
        loss_BC = self.loss_function(self.forward(x_BC), y_BC)
        return loss_BC

    # Loss PDE
    def lossPDE(self, x_PDE):
        x_min, x_max, t_min, t_max = self.domain[0], self.domain[1], self.domain[2], self.domain[3]
        x = x_PDE.clone()
        x.requires_grad = True  # Enable differentiation
        u = self.forward(x)
        u_x_t = autograd.grad(u, x, torch.ones([x.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]  # first derivative
        u_xx_tt = autograd.grad(u_x_t, x, torch.ones(x.shape).to(device), create_graph=True)[0]  # second derivative
        u_x = u_x_t[:, [0]]
        u_t = u_x_t[:, [1]]  # we select the 2nd element for t (the first one is x) (Remember the input X=[x,t])
        u_xx = u_xx_tt[:, [0]]  # we select the 1st element for x (the second one is t) (Remember the input X=[x,t])
        u_tt = u_xx_tt[:, [1]]

        # use dde instead of autograd.grad
        u_t = dde.grad.jacobian(u, x, i=0, j=1)  # Compute Jacobian matrix J: J[i][j] = dy_i / dx_j
        u_xx = dde.grad.hessian(u, x, i=0, j=0)  # Compute Hessian matrix H: H[i][j] = d^2y / dx_i dx_j

        if self.case == "example_1":
            f = torch.exp(-x[:, 1:]) * (torch.sin(np.pi * x[:, 0:1]) - np.pi**2 * torch.sin(np.pi * x[:, 0:1]))
            u = u_t - u_xx + f
        elif self.case == "example_2":
            zt = 2 * (x[:, 0:1] ** 2) * ((x[:, 0:1] - x_max) ** 2) * (x[:, 1:] - 1 / 2)
            zxx = (2 * ((x[:, 0:1] - x_max) ** 2) + 4 * x[:, 0:1] * (x[:, 0:1] - x_max) + 6 * (x[:, 0:1] ** 2) - 4 * x_max * x[:, 0:1]) * ((x[:, 1:] - 1 / 2) ** 2)
            source = zt - zxx
            u = u_t - u_xx - source

        u_hat = torch.zeros(x.shape[0], 1).to(device)
        loss_pde = self.loss_function(u, u_hat)
        return loss_pde

    def loss(self, x_BC, y_BC, x_PDE, w_bc, w_pde):
        loss_bc = self.lossBC(x_BC, y_BC)
        loss_pde = self.lossPDE(x_PDE)
        return w_bc * loss_bc + w_pde * loss_pde

    # The relative error is used to compare the test data.
    def error_l2_norm(self, x, y):
        error_u_predict = torch.norm(self.forward(x) - y, dim=1, p=2)
        error_u_predict_mean = error_u_predict.mean()
        return error_u_predict_mean, error_u_predict

    # The relative error is used to compare the test data.
    def relative_error_l2_norm(self, x, y):
        error_u_predict = torch.norm(self.forward(x) - y, dim=1, p=2)
        norm_u_real = torch.norm(y, dim=1, p=2)
        rela_error = error_u_predict / (norm_u_real)
        rela_error_mean = rela_error.mean()
        return rela_error_mean, rela_error

    # This function is used if you apply the L-BFGS Optimizer. For Adam is not necessary.
    def closure(self):
        optimizer.zero_grad()
        loss = self.loss(X_train_bc, Y_train_bc, X_train_Nf, w_bc=1, w_pde=1)
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            loss2 = self.lossBC(X_test, Y_test)
            # print("Training Error:", loss.detach().cpu().numpy(), "---Testing Error:", loss2.detach().cpu().numpy())
        return loss


def u_real(x, t, case):
    if case == "example_1":
        f = torch.exp(-t) * (torch.sin(np.pi * x))  # Example 1
    elif case == "example_2":
        f = (x**2) * ((x - 1) ** 2) * ((t - 1 / 2) ** 2)  # Example 2
    return f
