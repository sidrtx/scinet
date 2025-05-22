import deepxde as dde
import numpy as np
from deepxde.backend import tf

# Geometry and time
geom = dde.geometry.Rectangle([-1, -1], [1, 1])
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# Define PDE system
def pde_system(x, y):
    u = y[:, 0:1]
    phi = y[:, 1:2]
    vx = y[:, 2:3]
    vy = y[:, 3:4]

    # Derivatives
    u_t = dde.grad.jacobian(y, x, i=0, j=2)
    u_xx = dde.grad.hessian(y, x, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, i=0, j=1)

    phi_xx = dde.grad.hessian(y, x, i=1, j=0)
    phi_yy = dde.grad.hessian(y, x, i=1, j=1)

    vx_x = dde.grad.jacobian(y, x, i=2, j=0)
    vx_y = dde.grad.jacobian(y, x, i=2, j=1)
    vy_x = dde.grad.jacobian(y, x, i=3, j=0)
    vy_y = dde.grad.jacobian(y, x, i=3, j=1)

    # Parameters
    Du = 0.01
    alpha = 0.2
    usat = 1.0
    sigma = 1.0
    rho = u

    lam = 1.0
    mu = 0.5

    div_v = vx_x + vy_y
    eps_xx = vx_x
    eps_yy = vy_y
    eps_xy = 0.5 * (vx_y + vy_x)

    # Linear elasticity stress tensor divergence
    fx = (
        lam * tf.gradients(div_v, x)[0][:, 0:1]
        + 2 * mu * tf.gradients(eps_xx, x)[0][:, 0:1]
    )
    fy = (
        lam * tf.gradients(div_v, x)[0][:, 1:2]
        + 2 * mu * tf.gradients(eps_yy, x)[0][:, 1:2]
    )

    # PDEs
    eq_u = u_t - Du * (u_xx + u_yy) + alpha * u * (1 - u / usat)
    eq_phi = -(phi_xx + phi_yy) - rho
    eq_vx = fx
    eq_vy = fy

    return [eq_u, eq_phi, eq_vx, eq_vy]


# Initial and boundary conditions
def func_u(x):
    return np.exp(-5 * (x[:, 0:1] ** 2 + x[:, 1:2] ** 2))


ic = dde.IC(geomtime, func_u, lambda _, on_initial: on_initial, component=0)

bc_u = dde.DirichletBC(
    geomtime, lambda x: 0.0, lambda _, on_boundary: on_boundary, component=0
)
bc_phi = dde.DirichletBC(
    geomtime, lambda x: 0.0, lambda _, on_boundary: on_boundary, component=1
)
bc_vx = dde.DirichletBC(
    geomtime, lambda x: 0.0, lambda _, on_boundary: on_boundary, component=2
)
bc_vy = dde.DirichletBC(
    geomtime, lambda x: 0.0, lambda _, on_boundary: on_boundary, component=3
)

# Define data object
data = dde.data.TimePDE(
    geomtime,
    pde_system,
    [ic, bc_u, bc_phi, bc_vx, bc_vy],
    num_domain=10000,
    num_boundary=1000,
    num_initial=1000,
)

# Network
layer_size = [3] + [50] * 4 + [4]  # 3 inputs (x,y,t), 4 outputs (u,phi,vx,vy)
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=20000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
##########################################################################################################################
import matplotlib.pyplot as plt

# Create spatial grid
t_fixed = 0.5
nx = ny = 200
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
X, Y = np.meshgrid(x, y)
XYT = np.hstack(
    (X.flatten()[:, None], Y.flatten()[:, None], t_fixed * np.ones((nx * ny, 1)))
)

# Predict
pred = model.predict(XYT)
u_pred = pred[:, 0].reshape((nx, ny))
phi_pred = pred[:, 1].reshape((nx, ny))
vx_pred = pred[:, 2].reshape((nx, ny))
vy_pred = pred[:, 3].reshape((nx, ny))


# Plot utility
def plot_field(field, title):
    plt.figure(figsize=(5, 4))
    plt.contourf(X, Y, field, 100, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


plot_field(u_pred, "Nutrient Field u(x,y,t=0.5)")
plot_field(phi_pred, "Electric Potential φ(x,y,t=0.5)")
plot_field(np.sqrt(vx_pred**2 + vy_pred**2), "Displacement Magnitude ||v|| at t=0.5")


plt.figure(figsize=(5, 5))
plt.streamplot(
    x,
    y,
    vx_pred.T,
    vy_pred.T,
    color=np.sqrt(vx_pred.T**2 + vy_pred.T**2),
    cmap="plasma",
    density=1,
)
plt.title("Displacement Field v(x,y,t=0.5)")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="||v||")
plt.tight_layout()
plt.show()
