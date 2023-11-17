import torch
from utils.fcn_module import u_real
import numpy as np


class  data_gen():

    def __init__(self, domain, case):
        super().__init__()  # call __init__ from parent class
        self.case = case
        self.x_min = domain[0]
        self.x_max =  domain[1]
        self.t_min =  domain[2]
        self.t_max = domain[3]
        
        
    def get_test_dataset(self,N_test_x, N_test_t):
        x_min = self.x_min
        x_max = self.x_max
        t_min = self.t_min
        t_max = self.t_max
        
        x = torch.linspace(x_min, x_max, N_test_x).view(-1, 1)
        t = torch.linspace(t_min, t_max, N_test_t).view(-1, 1)

        # Create the mesh
        X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1))  # torch.Size([200, 200])

        # Evaluate real function
        y_real = u_real(X, T, self.case)  
        
        # Prepare Data 
        # Transform the mesh into a 2-column vector
        x_test = torch.hstack((X.transpose(1, 0).flatten().view(-1, 1), T.transpose(1, 0).flatten().view(-1, 1)))  # torch.Size([40000, 2])
        y_test = y_real.transpose(1, 0).flatten().view(-1, 1)  # Colum major Flatten (so we transpose it)
        return y_real, x_test, y_test, X, T 
    
    
    def get_PDE_dataset(self, N_train_x, N_train_t):
        """
        Generate a training set on the boundary that meets the boundary conditions    """

        # Training Data
        x_min = self.x_min
        x_max = self.x_max
        t_min = self.t_min
        t_max = self.t_max
        
        x = torch.linspace(x_min, x_max, N_train_x).view(-1, 1)
        t = torch.linspace(t_min, t_max, N_train_t).view(-1, 1)
        X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1))
        X_pde = torch.hstack((X.transpose(1, 0).flatten().view(-1, 1), T.transpose(1, 0).flatten().view(-1, 1)))
        return X_pde



    def get_BC_dataset(self,N_bc):
        
        x_min = self.x_min
        x_max = self.x_max
        t_min = self.t_min
        t_max = self.t_max
        case =  self.case

        """
        Generate a training set on the boundary that meets the boundary conditions    """

        # Training Data
        x_bc = torch.linspace(x_min, x_max, N_bc // 4).view(-1, 1) # We divide by 4 because the domain is a square
        t_bc = torch.linspace(t_min, t_max, N_bc // 4).view(-1, 1)
        X, T = torch.meshgrid(x_bc.squeeze(1), t_bc.squeeze(1))


        # Initial time data. Left Edge. min=<x=<xmax; t=0
        left_X = torch.hstack((X[:, 0][:, None], T[:, 0][:, None]))  # First column # The [:,None] is to give it                                                                      # the right dimension
        # Boundary data. 
        # Bottom Edge: x=min; tmin=<t=<max
        bottom_X = torch.hstack((X[0, :][:, None], T[0, :][:, None]))  # First row # The [:,None] is to give it
                                                                       # the right dimension
        # Top Edge: x=max; 0=<t=<1
        top_X = torch.hstack((X[-1, :][:, None], T[-1, :][:, None]))  # Last row # The [:,None] is to give it                                                                         # the right dimension

        # Boundary Conditions    
        # Bottom Edge: x=min; tmin=<t=<max
        bottom_Y = torch.zeros(bottom_X.shape[0], 1)
        # Top Edge: x=max; 0=<t=<1
        top_Y = torch.zeros(top_X.shape[0], 1)

        # Initial Condition
        if case == "example_1":    
            # Left_Y: y(x,0)=sin(pi*x)
            left_Y = torch.sin(np.pi * left_X[:, 0]).unsqueeze(1)

        elif case == "example_2":    
            # left_Y: y(x,0)=(x-1)**2*(x**2)/4
            left_Y = (((left_X[:, 0] - 1) ** 2) * (left_X[:, 0] ** 2) * 1 / 4).unsqueeze(1)


        X_train = torch.vstack([left_X, bottom_X, top_X])
        Y_train = torch.vstack([left_Y, bottom_Y, top_Y])


        return X_train, Y_train
