#################
# Libraries
#################
import logging
import os
import sys
import torch
import numpy as np
import time

from utils.plots import plot_mesh
from utils.utils import create_tests_folder
from utils.gen_plots import generate_gif
from utils.gen_data import data_gen
from utils.fcn_module import FCN
from pyDOE import lhs  # Latin Hypercube Sampling


#################
#  Device configuration
#################

# Set default dtype to float32
torch.set_default_dtype(torch.float)

# Set the device, if CUDA is installed we use the GPU, otherwise we use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if device == "cuda":
    print(torch.cuda.get_device_name())

#################
# Tunning Parameters
#################

steps = 3000  # Number of steps in the optimization
batch_size = 1000  # Batch size in the optimization
w_bc = 100  # Extra weights for the loss_bc functional
w_pde = 1  # Extra weights for the loss_pde functional
lr = 1e-3  # Learning rate for the stochastic gradient descent
layers = np.array([2, 50, 50, 50, 50, 1])  # Number of neurons on each layer of the fully connected neural network
log_freq = 50  # log frequency

## Choose the example. Each example has different domain and sourse term.
case = "example_1"
if case == "example_1":
    domain = [-1, 1, 0, 1]
elif case == "example_2":
    domain = [0, 1, 0, 10]

# Define the number of training/test data
N_test_x = 200  #  Number of testing points in space @@@ actually
N_test_t = 200  #  Number of testing points in time
N_train_x = 150  #  Number of training points in space
N_train_t = 150  #  Number of training points in time
N_bc = 1000  #  Number of training points on the boundary

# Create a folder with the results
test_folder = create_tests_folder(parent_folder="results", prefix=f"_{case}")
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(test_folder + "/0_log.txt")], level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

#################
# Generate data
#################

DG = data_gen(domain, case)
x_train_pde = DG.get_PDE_dataset(N_train_x, N_train_t)
x_train_bc, y_train_bc = DG.get_BC_dataset(N_bc)  # Generate the train dataset (time and space)
y_real, x_test, y_test, X, T = DG.get_test_dataset(N_test_x, N_test_t)

##########
# Is it possible to use Latin Hypercube Sampling instead of the get_PDE_dataset function
# lb = x_test[0]  # first value
# ub = x_test[-1]  # last value
# x_train_pde = lb + (ub - lb) * lhs(2, N_train_x*N_test_t)  # Choose 20000 points, 2 as the inputs are x and t
##########


# If the device is CUDA (CUDA is installed), the data are sent to the GPU
x_train_bc = x_train_bc.float().to(device)  # Training Points (BC)
y_train_bc = y_train_bc.float().to(device)  # Training Points (BC)
x_train_pde = x_train_pde.float().to(device)  # Collocation Points
X_test = x_test.float().to(device)  # the input dataset (complete)
Y_test = y_test.float().to(device)  # the real solution

#################
# Create Model and Optimazer
#################

PINN = FCN(layers, case, domain)
PINN.to(device)
print(PINN)
optimizer = torch.optim.Adam(PINN.parameters(), lr=lr, amsgrad=False)

# It is possible to consider another optimazer
# L-BFGS Optimizer
# optimizer = torch.optim.LBFGS(PINN.parameters(), lr=lr,
#                               max_iter = steps,
#                               max_eval = None,
#                               tolerance_grad = 1e-05,
#                               tolerance_change = 1e-09,
#                               history_size = 100,
#                               line_search_fn = 'strong_wolfe')

#################
# Training process
#################

# Creating a root directory to store information
res_dict = {"loss": [], "loss_bc": [], "loss_pde": [], "rela_err_l2": []}
start_time_a = time.time()
count = 1
plot_mesh(X, T, y_real, case, name=f"0_real", error=0, folder=test_folder)  # plot real solution
for i in range(1, steps + 1):
    idx = np.random.choice(x_train_pde.shape[0], batch_size, replace=False)  # Choose randomly a batch in x_train_pde
    loss = PINN.loss(x_train_bc, y_train_bc, x_train_pde[idx, :], w_bc=w_bc, w_pde=w_pde)  # use mean squared error
    optimizer.zero_grad()  # We 'clean' the optimazer
    loss.backward()  # Backpropagation (compute the gradients)
    optimizer.step()  # Actualization of the parameters
    # optimizer.step(PINN.closure) # L-BFGS Optimizer

    # Create the plots with the solution with the current parameters, and the relative error.
    res_dict["loss"].append(loss.item())
    with torch.no_grad():
        err_l2_mean, error = PINN.error_l2_norm(X_test, Y_test)
        res_dict["rela_err_l2"].append(err_l2_mean.item())
    if i % log_freq == 0 or i == 1:
        u_predict = PINN(X_test)
        arr_y1 = u_predict.reshape(shape=[N_test_t, N_test_x]).transpose(1, 0).detach().cpu()
        plot_mesh(X, T, arr_y1, case, name=f"approx_{count}", loss=loss.item(), folder=test_folder, iter=i)
        # error = error.reshape(shape=[N_test_t, N_test_x]).transpose(1, 0).detach()
        # plot_mesh(X, T, error, name=f"error_{count}", error=err_l2_mean, folder=test_folder, iter=i)
        if i % (log_freq*2) == 0 or i == 1:
            logger.info(f"| Iter: {i} | Loss: {loss.item():.4f} | Total_time: {(time.time() - start_time_a)/60:.1f}min")
        count += 1
    # path = test_folder + "/A_results_dict.npy"
    # np.save(path, np.asarray(res_dict, dtype=object))
print('Finished training')
#################
# Animation
#################

# Generate animations showing the convergence to the real solution, and the relative error
## It is necessary to have the imageio_ffmpeg  (pip install imageio_ffmpeg)
generate_gif(test_folder, count, remove_imgs=True)
