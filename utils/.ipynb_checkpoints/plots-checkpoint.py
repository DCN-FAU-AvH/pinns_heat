# auxiliary functions for plotting

from matplotlib import gridspec, pyplot as plt


def plot_mesh(x, t, y, case, folder="", name="", show=None, error=0, loss=0, iter=0):
    X, T = x.cpu(), t.cpu()
    F_xt = y.cpu()

    fig, ax = plt.subplots(1, 1, dpi=200)
    cp = ax.contourf(X, T, F_xt, 20, cmap="rainbow")
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_xlabel("x")
    # ax.set_ylabel("x2")
    ax.set_ylabel("t")
    if error > 0 and loss != 0:
        ax.set_title(f"Iteration: {iter}, Loss: {loss:.5f} \n  Relative error: {error:.5f}")
    elif error > 0:
        ax.set_title(f"Iteration: {iter} \n Relative error: {error:.5f}")
    elif loss != 0:
        ax.set_title(f"Iteration: {iter}, Loss: {loss:.5f}")
    else:
        ax.set_title(f"Iteration: {iter}")
    plt.savefig(f"{folder}/contour_{name}.png")  # Save the first image as an image file
    plt.close()

    plt.figure(dpi=200)
    ax = plt.axes(projection="3d")
    ax.plot_surface(X.numpy(), T.numpy(), F_xt.numpy(), cmap="rainbow")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    
    if error > 0 and loss != 0:
        ax.set_title(f"Iteration: {iter}, Loss: {loss:.5f} \n  Relative error: {error:.5f}")
    elif error > 0:
        ax.set_title(f"Iteration: {iter} \n Relative error: {error:.5f}")
    elif loss != 0:
        ax.set_title(f"Iteration: {iter}, Loss: {loss:.5f}")
    else:
        ax.set_title(f"Iteration: {iter}")
        
    if case=='example_1':
        ax.set_zlim3d(-1, 1) 
    else:
        ax.set_zlim3d(0, 6) 
        
    plt.savefig(f"{folder}/{name}.png")  # Save the first image as an image file
    if show:
        plt.show()
    plt.close()
