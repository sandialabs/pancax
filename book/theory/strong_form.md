# Strong Form
Consider a general PDE defined in the strong form as follows
$$
\mathbf{A}\frac{\partial^2 \mathbf{u}}{\partial t^2} + \mathbf{B}\frac{\partial \mathbf{u}}{\partial t} + \mathcal{L}\left[\mathbf{u}\left(\mathbf{x}, t\right)\right] = 0,
$$
where $\mathbf{u}$ is an arbitrary solution field (potentially vector or tensor valued) and $\mathcal{L}$ is a differential operator. To fully close an initial boundary value problem (IBVP) we also need appropriate initial and boundary conditions. We can write this generally as
$$
\mathbf{u}\left(\mathbf{x}, 0\right) = f\left(\mathbf{x}\right),
$$
$$
\dot{\mathbf{u}}\left(\mathbf{x}, 0\right) = g\left(\mathbf{x}\right),
$$
$$
\mathbf{u}\left(\mathbf{x}, t\right) = h\left(\mathbf{x}\right) \quad\text{for}\quad\mathbf{x}\in\partial\Omega_\mathbf{u},
$$
$$
a\mathbf{u}\left(\mathbf{x}, t\right) + b\nabla\cdot\mathbf{u}\left(\mathbf{x}, t\right) = h\left(\mathbf{x}\right) \quad\text{for}\quad\mathbf{x}\in\partial\Omega_\mathbf{u},
$$
where the above represent general initial conditions on the solution field and time derivative and general boundary conditions on the the solution field in the form of general Dirichlet BCs and general Robin/Neumann BCs.

Our goal in pancax is to seek approximate solutions to the above general equation via physics informed neural networks (PINNs). We can do this in a number of ways, but the simplest and most classical is the following approach. We approximate the solution field with a multi-layer perceptron (MLP)
$$
\mathbf{u}\left(\mathbf{x}, t\right) \approx \mathcal{N}\left(\mathbf{x}, t\right).
$$
With this in hand, we can now take derivatives with respect to $\mathbf{x}$ or $t$ to obtain the necessary derivatives in our operator. For example
$$
\frac{\partial \mathbf{u}}{\partial t} \approx \frac{\partial \mathcal{N}\left(\mathbf{x}, t\right)}{\partial t},
$$
which put another way is saying to differentiate the outputs of the neural network with respect to a subset of some of it's inputs. This can be achieved in any standard modern neural network library through automatic differentiation. However, this is not always an optimal approach in terms of both runtime and numerical accuracy, especially when it comes to nested differentiation.

Other derivatives can re written similarly and some are given below
$$
\frac{\partial^2 \mathbf{u}}{\partial t^2} \approx \frac{\partial^2 \mathcal{N}\left(\mathbf{x}, t\right)}{\partial t^2},
$$
$$
\nabla\mathbf{u} \approx \frac{\partial \mathcal{N}\left(\mathbf{x}, t\right)}{\partial \mathbf{x}},
$$
$$
\nabla\cdot\mathbf{u} \approx \frac{\partial \mathcal{N}\left(\mathbf{x}, t\right)}{\partial \mathbf{x}}:\mathbf{I}.
$$

One of the more common examples encountered in the PINN literature is the Burger's equation. This is given by
$$
\frac{\partial u}{\partial t} + u\cdot\nabla u = 0
$$