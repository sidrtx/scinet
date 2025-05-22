import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from deepxde.backend import tf

# Geometry and time domain
L = 1.0
geom = dde.geometry.Rectangle([0, 0], [L, L])
timedomain = dde.geometry.TimeDomain(0, 2)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Parameters
Du = 0.01
eps = 1e-2
kappa = 1.0
mu0 = 0.1
K = 0.2


# PDE system definition
def pde(x, y):
    u, c = y[:, 0:1], y[:, 1:2]

    u_t = dde.grad.jacobian(y, x, i=0, j=2)
    c_t = dde.grad.jacobian(y, x, i=1, j=2)

    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0) + dde.grad.hessian(
        y, x, component=0, i=1, j=1
    )
    c_xx = dde.grad.hessian(y, x, component=1, i=0, j=0) + dde.grad.hessian(
        y, x, component=1, i=1, j=1
    )

    mu = mu0 * c / (K + c)
    return [u_t - Du * u_xx - mu * u * (1 - u), c_t - eps * c_xx + kappa * u * c]


# Initial conditions
def ic_u(x):
    return np.exp(-50 * ((x[:, 0:1] - 0.5) ** 2 + (x[:, 1:2] - 0.5) ** 2))


def ic_c(x):
    return np.ones_like(x[:, 0:1])


# Conditions and boundary definitions
ic1 = dde.IC(geomtime, ic_u, lambda _, on_initial: on_initial, component=0)
ic2 = dde.IC(geomtime, ic_c, lambda _, on_initial: on_initial, component=1)
bc1 = dde.NeumannBC(
    geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component=0
)
bc2 = dde.NeumannBC(
    geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component=1
)

# Data definition
data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic1, ic2, bc1, bc2],
    num_domain=5000,
    num_boundary=1000,
    num_initial=500,
    solution=None,
    num_test=1000,
)

# Network architecture
net = dde.maps.FNN([3] + [64] * 3 + [2], "tanh", "Glorot normal")
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=1000)

# Predict at t=2 to visualize tumor and nutrient distribution
x, y = geomtime.uniform_points(10000, True), None
y_pred = model.predict(x)

# Filter for t = 2
mask = np.isclose(x[:, 2], 2.0, atol=0.01)
x_plot = x[mask]
u_plot = y_pred[mask, 0]
c_plot = y_pred[mask, 1]

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.tricontourf(x_plot[:, 0], x_plot[:, 1], u_plot, 100, cmap="viridis")
plt.colorbar()
plt.title("Tumor Cell Density u(x, y, t=2)")

plt.subplot(1, 2, 2)
plt.tricontourf(x_plot[:, 0], x_plot[:, 1], c_plot, 100, cmap="plasma")
plt.colorbar()
plt.title("Nutrient Concentration c(x, y, t=2)")

plt.tight_layout()
plt.show()
