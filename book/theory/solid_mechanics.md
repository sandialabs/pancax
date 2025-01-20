# Solid Mechanics
The original motivator for writing pancax was for applications in solid mechanics, specifically for leveraging digital image correlation data in inverse problems in PINNs.

The best approach to work with solid mechanics is to build off of an approach known in the PINNs community as the "Deep Energy Method" (DEM) which is really just a variational principle in solid mechanics with neural network approximations for the solution field.

We can write this variational principle as follows

$$
\Pi\left(\mathbf{u}, \nabla\mathbf{u}\right) = \int_\Omega\psi\left(\nabla\mathbf{u}\right)d\Omega 
                                             - \int_{\partial\Omega}\mathbf{t}\cdot\mathbf{u}dS 
                                             - \int_\Omega\mathbf{b}\cdot\mathbf{u}d\Omega.
$$
The first variation of this gives rise to the principle of virtual work
$$
\delta\Pi = \frac{\partial\Pi}{\partial\mathbf{u}}\cdot\delta\mathbf{u} 
          + \frac{\partial\Pi}{\partial\nabla\mathbf{u}}\cdot\delta\nabla\mathbf{u}
$$
which can be simplified to 
$$
\delta\Pi = -\int_{\partial\Omega}\mathbf{t}\cdot\delta\mathbf{u}dS 
            - \int_\Omega\mathbf{b}\cdot\delta\mathbf{u}d\Omega
            + \int_\Omega\frac{\partial\psi}{\partial\nabla\mathbf{u}}:\delta\nabla\mathbf{u} d\Omega,
$$
and using the identities $\delta\nabla\mathbf{u} = \delta\mathbf{F}$ where 
$\mathbf{F} = \nabla\mathbf{u} + \mathbf{I}$ and
$\mathbf{P} = \frac{\partial\psi}{\partial\mathbf{F}}$
we can re-write the above as
$$
\delta\Pi = \int_\Omega\mathbf{P}:\delta\mathbf{F} d\Omega
            -\int_{\partial\Omega}\mathbf{t}\cdot\delta\mathbf{u}dS 
            - \int_\Omega\mathbf{b}\cdot\delta\mathbf{u}d\Omega.
$$
Using either of the above forms gives rise to traction boundary condition enforcement by construction.