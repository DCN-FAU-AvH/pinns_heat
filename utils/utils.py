# auxiliary functions for plotting

import os
import time
from matplotlib import gridspec, pyplot as plt
import numpy as np
import pytz
import torch


def create_tests_folder(parent_folder="", prefix="", postfix=""):
    """
    Create tests_folder based on time and change it to Berlin time zone
    """
    time_stamp = int(time.time())
    time_zone = pytz.timezone("Europe/Berlin")
    test_time = pytz.datetime.datetime.fromtimestamp(time_stamp, time_zone)
    test_time = test_time.strftime("%Y%m%d-%H%M%S")
    tests_folder = os.getcwd() + f"/{parent_folder}/test{prefix}_{test_time}{postfix}"
    os.makedirs(tests_folder)
    print(f"\nWorking in folder {tests_folder}\n")
    return tests_folder


def get_PDE_dataset(domain, N_train_x, N_train_t):
    """
    Generate a training set on the boundary that meets the boundary conditions
    """
    # Training Data
    x_min, x_max, t_min, t_max = domain[0], domain[1], domain[2], domain[3]
    x = torch.linspace(x_min, x_max, N_train_x).view(-1, 1)
    t = torch.linspace(t_min, t_max, N_train_t).view(-1, 1)
    X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1))
    X_pde = torch.hstack((X.transpose(1, 0).flatten().view(-1, 1), T.transpose(1, 0).flatten().view(-1, 1)))
    return X_pde


def get_BC_dataset(domain, N_bc, case):
    """
    Generate a training set on the boundary that meets the boundary conditions
    """
    # Training Data
    x_min, x_max, t_min, t_max = domain[0], domain[1], domain[2], domain[3]
    x_bc = torch.linspace(x_min, x_max, N_bc // 4).view(-1, 1)  # We divide by 4 because the domain is a square
    t_bc = torch.linspace(t_min, t_max, N_bc // 4).view(-1, 1)
    X, T = torch.meshgrid(x_bc.squeeze(1), t_bc.squeeze(1))
    # Initial time data. Left Edge. min=<x=<xmax; t=0
    left_X = torch.hstack((X[:, 0][:, None], T[:, 0][:, None]))  # First column # The [:,None] is to give it the right dimension
    # Boundary data.
    # Bottom Edge: x=min; tmin=<t=<max
    bottom_X = torch.hstack((X[0, :][:, None], T[0, :][:, None]))  # First row # The [:,None] is to give it
    # the right dimension
    # Top Edge: x=max; 0=<t=<1
    top_X = torch.hstack((X[-1, :][:, None], T[-1, :][:, None]))  # Last row # The [:,None] is to give it the right dimension
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
