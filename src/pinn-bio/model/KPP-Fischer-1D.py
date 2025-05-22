import deepxde as dde
import numpy as np
from deepxde.backend import tf
import matplotlib.pyplot as plt

# 1. Problem parameters
D = 0.01  # diffusion coefficient
r = 1.0  # intrinsic growth rate
K = 1.0  # carrying capacity
m0 = 0.3  # mean Allee threshold
A = 0.1  # amplitude of oscillation
T = 1.0  # period of oscillation

# 2. Geometry: 1D interval × time
geom = dde.geometry.Interval(0.0, 1.0)
timedomain = dde.geometry.TimeDomain(0.0, 1.0)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# 3. PDE residual
def pde(x, y):
    # x[:, 0:1] = spatial coordinate; x[:, 1:2] = time
    u = y[:, 0:1]
    u_t = dde.grad.jacobian(y, x, i=0, j=1)
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    # time‑dependent Allee threshold
    m_t = m0 + A * tf.sin(2 * np.pi * x[:, 1:2] / T)
    # reaction term with Allee effect
    reaction = r * u * (1 - u / K) * (u - m_t)
    return u_t - D * u_xx - reaction


# 4. Initial condition: Gaussian bump centered at x=0.5
def initial_u(x):
    return 0.6 * np.exp(-(((x[:, 0:1] - 0.5) / 0.1) ** 2))


ic = dde.IC(
    geomtime,
    initial_u,
    lambda _, on_init: on_init,
    component=0,
)


# 5. Neumann boundary condition (zero flux) at x=0 and x=1
def boundary(_, on_boundary):
    return on_boundary


bc = dde.NeumannBC(
    geomtime,
    lambda x: 0.0,
    boundary,
    component=0,
)

# 6. Assemble the data
data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic, bc],
    num_domain=20000,
    num_boundary=2000,
    num_initial=2000,
    num_test=5000,
)

# 7. Neural network
layer_size = [2] + [50] * 4 + [1]  # inputs: (x, t); output: u
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# 8. Model compilation and training (no analytic solution → no metrics)
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=10000)

# 9. Visualization: u(x, t) at t = 0.0, 0.5, 1.0
xx = np.linspace(0, 1, 200)
for t_val in [0.0, 0.5, 1.0]:
    X = np.vstack((xx, np.full_like(xx, t_val))).T
    u_pred = model.predict(X).flatten()
    plt.plot(xx, u_pred, label=f"t = {t_val:.1f}")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Fisher–KPP with Time‑Dependent Allee Effect")
plt.legend()
plt.show()
