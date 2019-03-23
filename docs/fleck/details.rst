**********************
Implementation details
**********************

Vectorized rotational modulation computation
--------------------------------------------

Suppose we have :math:`N` stars, each with :math:`M` starspots, distributed
randomly above minimum latitudes :math:`{\bf \ell_{min}}`, observed at :math:`N`
inclinations :math:`\vec{i}_\star` (one unique inclination per star), observed at
:math:`O` phases throughout a full rotation
:math:`\vec{\phi} \sim \mathcal{U}(0, 90^\circ)`.

We initialize each star such that its rotation axis is aligned with the
:math:`\hat{z}` axis, and set the observer at :math:`x \rightarrow \infty`,
viewing down the :math:`\hat{x}` axis towards the origin.


We define the rotation matrices about the :math:`\hat{y}` and :math:`\hat{z}`
axes for a rotation by angle :math:`\theta`:

.. math::

    \begin{eqnarray}
    {\bf R_y}(\theta) &= \begin{bmatrix}
    \cos \theta & 0 & \sin \theta \\
    0 & 1 & 0 \\
    -\sin \theta & 0 & \cos \theta \\
    \end{bmatrix} \\
    {\bf R_z}(\theta) &= \begin{bmatrix}
    \cos \theta &  -\sin \theta & 0 \\
    \sin \theta &   \cos \theta & 0\\
    0 & 0 & 1\\
    \end{bmatrix}
    \end{eqnarray}

We begin with the matrix of starspot positions in Cartesian coordinates
:math:`{\bf C_i}`,

.. math::

    {\bf C_i}=
      \begin{bmatrix}
        x_1 & y_1 & z_1 \\
        x_2 & y_2 & z_2 \\
        \vdots  & \vdots  & \vdots \\
        x_M & y_M & z_M
      \end{bmatrix}


for :math:`i=1` to :math:`N` with shape :math:`(3, M)`, which we collect into
the array :math:`{\bf S}`,

.. math::

    {\bf S}=
      \begin{bmatrix}
        {\bf C_1}, & {\bf C_2}, & \dots, & {\bf C_N}
      \end{bmatrix}

of shape :math:`(3, M, N)`. We rotate the starspot positions through each angle
in :math:`\phi_j` for :math:`j=1` to :math:`O` by multiplying :math:`\bf S` by
the rotation array

.. math::

    {\bf R_z}=
      \begin{bmatrix}
        [[{\bf R_z}(\phi_1)]], & [[{\bf R_z}(\phi_2)]], & \dots, & [[{\bf R_z}(\phi_O)]]
      \end{bmatrix}

with shape :math:`(O, 1, 1, 3, 3)`. Using Einstein notation, we transform the
Cartesian coordinates array :math:`\bf C` with:

.. math::

    \begin{equation}
        {\bf R_z}_{[lm\dots]ij} {\bf S}^{j[lm\dots]} = {\bf S^\prime}_{i[lm\dots]}
    \end{equation}

to produce a array with shape :math:`(3, O, M, N)`, where :math:`lm...`
indicates an optional additional set of dimensions. Then after each star has
been rotated about its rotation axis in :math:`\hat{z}`, we rotate each star
about the :math:`\hat{y}` axis to represent the stellar inclinations
:math:`i_{\star, k}` for :math:`k=1` to :math:`N`, using the rotation array

.. math::

    {\bf R_y}=
      \begin{bmatrix}
        {\bf R_y}(i_{\star, 1}), & {\bf R_y}(i_{\star, 2}), & \dots, & {\bf R_y}(i_{\star, N})
      \end{bmatrix}

with shape :math:`(N, 3, 3)`, by doing

.. math::

    \begin{equation}
        {\bf R_y}_{[lm\dots]ij} {\bf S^\prime}^{j[lm\dots]} = {\bf S^{\prime\prime}}_{i[lm\dots]}
    \end{equation}

which produces another array of shape :math:`(3, O, M, N)`. Now we extract the
second and third axes of the first dimension, which correspond to the :math:`y`
and :math:`z` (sky-plane) coordinates, and compute the radial position of the
starspot :math:`{\bf \rho} = \sqrt{y^2 + z^2}`, where :math:`\bf \rho` has shape
:math:`(O, M, N)`. We now mask the array so that any spots with :math:`x < 0`
are masked from further computations, as these spots will not be visible to the
observer. We use :math:`\rho` to compute the quadratic limb darkening

.. math::

    \begin{equation}
        I(\rho) = \frac{1}{\pi} \frac{1 - u_1 (1 - \mu) - u_2 (1 - \mu)^2}{1 - u_1/3 - u_2/6}
    \end{equation}

for :math:`\mu = \sqrt{1 - \rho^2}`. We compute the flux missing due to
starspots of radii :math:`\bf R_{\rm spot}`, which has shape :math:`(M, N)`:

.. math::

    \begin{equation}
    {\rm F_{spots}} = \pi {\bf R}_{\rm spot}^2  (1 - c) \frac{I(r)}{I(0)} \sqrt{1 - {\bf \rho}^2}
    \end{equation}

The unspotted flux of the star is

.. math::

    \begin{equation}
        {\rm F_{unspotted}} = \int_0^R 2\pi r I(r) dr,
    \end{equation}

so the spotted flux is

.. math::

    \begin{equation}
        {\rm F_{spotted}} = 1 - \frac{\rm F_{spots,ijk}F_{spots}^{ik}}{\rm F_{unspotted}}
    \end{equation}

Limitations of the model
------------------------
The model presented above works best for spots that are small. The array masking
step for :math:`x < 0` does not account for the change in stellar flux due to
large starspots which straddle the limb of the star. Large starspots also have
differential limb-darkening across their extent, which is not computed by
fleck.