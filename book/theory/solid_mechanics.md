# Solid Mechanics
The original motivator for writing pancax was for applications in solid mechanics, specifically for leveraging digital image correlation data in inverse problems in PINNs.

The best approach to work with solid mechanics is to build off of an approach known in the PINNs community as the "Deep Energy Method" (DEM) which is really just a variational principle in solid mechanics with neural network approximations for the solution field.

We can write this variational principle as follows

$$
\Pi\left(\mathbf{u}, \nabla\mathbf{u}\right) = \int_\Omega\psi\left(\nabla\mathbf{u}\right)d\Omega - \int_{\partial\Omega}\mathbf{t}\cdot\mathbf{u}dS - \int_\Omega\mathbf{b}\cdot\mathbf{u}d\Omega.
$$
The first variation of this gives rise to the principle of virtual work
$$
\delta\Pi = \frac{\partial\Pi}{\partial\mathbf{u}}\cdot\delta\mathbf{u} + \frac{\partial\Pi}{\partial\nabla\mathbf{u}}\cdot\delta\nabla\mathbf{u}
$$
