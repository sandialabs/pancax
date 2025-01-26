# Collocation PINNs
Collocation PINNs are the most commonly encountered type of PINN in the literuate. The main concept is there is an unconnected point cloud (e.g. meshless) of inputs $\{\mathbf{x}_i, t_i\}$ to a neural network $\mathcal{N}\left(\mathbf{x}, t\right)$. The goal is to minimize the residual of the strong form of the governing equation of interest via standard neural network optimizers such as stochastic gradient descent, Adam, etc.

Let's consider a general equation of the form 
$$
\mathbf{A}\frac{\partial^2 \mathbf{u}}{\partial t^2} + \mathbf{B}\frac{\partial \mathbf{u}}{\partial t} + \mathcal{O}\left[\mathbf{u}\left(\mathbf{x}, t\right)\right] = 0,
$$
and define its residual to be
$$
\mathbf{R} = \mathbf{A}\frac{\partial^2 \mathbf{u}}{\partial t^2} + \mathbf{B}\frac{\partial \mathbf{u}}{\partial t} + \mathcal{O}\left[\mathbf{u}\left(\mathbf{x}, t\right)\right],
$$
and approximate the solution $\mathbf{u}$ by $\mathbf{u}_i \approx \mathcal{N}\left(\mathbf{x}_i, t_i\right)$.

We can define a loss function for the residual as follow
$$
\mathcal{L}_r = \sum_i \|\mathbf{R}_i\|^2
$$
which can be expanded to
$$
\mathcal{L}_r = \sum_i \|\mathbf{A}\frac{\partial^2 \mathcal{N}}{\partial t^2} + \mathbf{B}\frac{\partial \mathcal{N}}{\partial t} + \mathcal{O}\left[\mathcal{N}\left(\mathbf{x}, t\right)\right]\|^2
$$

Regarding boundary conditions (BCs), the most common choice (although usually incorrect) is to weakly enforce BCs via additional loss function terms. Alternatively, signed distance functions (SDFs) can be used to exactly enforce Dirichlet and Neumann BCs by imposing structure on the solution space.

## Sub-optimal Enforcement of BCs
For the loss function approach, terms for different BC types are as follows
$$
\mathcal{L}_{\mathbf{u},IC} = \sum_i\|\mathbf{u}\left(\mathbf{x}_i, 0\right) - f\left(\mathbf{x}_i\right)\|^2,
$$
$$
\mathcal{L}_{\dot{\mathbf{u}},IC} = \sum_i\|\dot{\mathbf{u}}_i\left(\mathbf{x}_i, 0\right) - g\left(\mathbf{x}_i\right)\|^2,
$$
$$
\mathcal{L}_{\mathbf{u},BC} = \sum_i\|\mathbf{u}_i\left(\mathbf{x}_i, t_i\right) - h\left(\mathbf{x}_i, t_i\right)\|^2 \quad\text{for}\quad\mathbf{x}_i\in\partial\Omega_\mathbf{u}\quad\text{and}\quad t_i\in\left[0,T\right],
$$
$$
\mathcal{L}_{\nabla\mathbf{u},BC} = \sum_i\|a\mathbf{u}_i\left(\mathbf{x}_i, t_i\right) + b\mathbf{n}_i\cdot\nabla\mathbf{u}_i\left(\mathbf{x}_i, t_i\right) - m\left(\mathbf{x}_i, t_i\right)\|^2 \quad\text{for}\quad\mathbf{x}_i\in\partial\Omega_\mathbf{u}\quad\text{and}\quad t_i\in\left[0,T\right],
$$

The total loss function is then
$$
\mathcal{L} = w_r\mathcal{L}_r + w_{\mathbf{u},IC}\mathcal{L}_{\mathbf{u},IC} + w_{\mathbf{\dot{u}},IC}\mathcal{L}_{\dot{\mathbf{u}},IC} + w_{\mathbf{u},BC}\mathcal{L}_{\mathbf{u},BC} + w_{\nabla\mathbf{u},BC}\mathcal{L}_{\nabla\mathbf{u},BC}
$$

## Optimal Enforcement of BCs
For Dirchlet BCs
$$
\mathbf{u}\left(\mathbf{x}, t\right) = f\left(\mathbf{x}\right) + s\left(\mathbf{x}\right)\mathcal{N}\left(\mathbf{x}, t\right),
$$
where $s$ is a function constructed such that $s\left(\mathbf{x}\right) = 0$ on $\partial\Omega_\mathbf{u}$. This ensures enforcement of Dirichlet BCs by construction and alleviates the burden of learning on the network for this relationship.

More complex relationships are also possible for enforcing Neumann and Robin BCs TODO right this down.

In this case, the total loss function is greatly simplified to just the residual term
$$
\mathcal{L} = \mathcal{L}_r
$$