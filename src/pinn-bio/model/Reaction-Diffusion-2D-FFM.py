import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from deepxde.backend import tf

# Set random seed for reproducibility
dde.config.set_random_seed(42)

# Parameters for the Schnakenberg model
Du = 1e-5
Dv = 1e-3
a = 0.1
b = 0.9


# Fourier feature mapping for high-frequency representation
def fourier_features(x, num_features=256):
    """Generate Fourier features for input x."""
    L = 1.0  # Domain length
    scale = 2 * np.pi * np.linspace(1, num_features, num_features) / L
    return np.concatenate([np.sin(scale * x), np.cos(scale * x)], axis=-1)


# Define the neural network with Fourier feature input
def neural_network(x):
    """Define the neural network architecture."""
    x = fourier_features(x)
    net = dde.nn.FNN(
        [x.shape[1]] + [128] * 4 + [2],
        activation="tanh",
        kernel_initializer="Glorot uniform",
    )
    return net(x)


# Define the PDE system (Schnakenberg model)
def pde(x, y):
    """Define the PDE system."""
    u, v = y[:, 0:1], y[:, 1:2]
    u_t = dde.grad.jacobian(y, x, i=0, j=2)
    v_t = dde.grad.jacobian(y, x, i=1, j=2)
    u_xx = dde.grad.hessian(y, x, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, i=0, j=1)
    v_xx = dde.grad.hessian(y, x, i=1, j=0)
    v_yy = dde.grad.hessian(y, x, i=1, j=1)
    return [
        u_t - Du * (u_xx + u_yy) - (a - u + u**2 * v),
        v_t - Dv * (v_xx + v_yy) - (b - u**2 * v),
    ]


# Define the spatial and temporal domains
geom = dde.geometry.Rectangle([0, 0], [1, 1])
timedomain = dde.geometry.TimeDomain(0, 10)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# Initial conditions with small random perturbations
def initial_u(x):
    return a + b + 0.01 * np.random.rand(*x[:, 0:1].shape)


def initial_v(x):
    return b / (a + b) ** 2 + 0.01 * np.random.rand(*x[:, 0:1].shape)


ic_u = dde.icbc.IC(geomtime, initial_u, lambda x, on_initial: on_initial, component=0)
ic_v = dde.icbc.IC(geomtime, initial_v, lambda x, on_initial: on_initial, component=1)

# Neumann (zero-flux) boundary conditions
bc_u = dde.icbc.NeumannBC(
    geomtime, lambda x: 0, lambda x, on_boundary: on_boundary, component=0
)
bc_v = dde.icbc.NeumannBC(
    geomtime, lambda x: 0, lambda x, on_boundary: on_boundary, component=1
)

# Define the data
data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic_u, ic_v, bc_u, bc_v],
    num_domain=40000,
    num_boundary=2000,
    num_initial=2000,
    num_test=1000,
)

# Define the model
net = dde.nn.FNN(
    [3] + [128] * 4 + [2], activation="tanh", kernel_initializer="Glorot uniform"
)
model = dde.Model(data, net)

# Compile and train the model
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=10000)

# Optional: Fine-tune with L-BFGS optimizer
model.compile("L-BFGS")
losshistory, train_state = model.train()

# Save and plot the results
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


# Function to plot the solution at a specific time snapshot
def plot_snapshot(model, t_snap=5.0, resolution=200):
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    T = np.full_like(X, t_snap)
    input_points = np.vstack((X.flatten(), Y.flatten(), T.flatten())).T
    pred = model.predict(input_points)
    u = pred[:, 0].reshape(resolution, resolution)
    v = pred[:, 1].reshape(resolution, resolution)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"U at t={t_snap}")
    plt.contourf(X, Y, u, levels=100, cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title(f"V at t={t_snap}")
    plt.contourf(X, Y, v, levels=100, cmap="viridis")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# Plot the solution at t = 5.0
plot_snapshot(model, t_snap=5.0)
