# RNN equation with $\tau=1, R=1, I_\textrm {ext}(t) = 0$

$N$ neurons, each with potential $h_i(t), i = 1,\cdots, N$ evolve according to

$$
\dot h_i(t) = -h_i(t) + \sum_{j=1}^{N} J_{ij} \phi(h_j(t))
$$

where

$$
J_{ij} = \frac 1N \sum_{\mu=1}^p \xi_{\mu,i} \tilde \phi(\xi_{\mu,j}),
\quad
\xi_{\mu,i} \sim \mathcal{N}(0,1),
\quad
\tilde \phi(\xi) = \frac{\phi(\xi) - \mathrm{E}[\phi(\xi)]}{\mathrm{Var}[\phi(\xi)]}
$$

Overlaps are defined as

$$
m_\mu(t) = \frac 1N \sum_{i=1}^N \tilde \phi(\xi_{\mu,i}) \phi(h_i(t))
$$

Some fixed points :

- $h_i = \xi_{\mu,i} \quad \forall \mu = 1,\cdots,p$
- $h_i=0$

# Neural field equation in $\mathbb R^p$

Neurons are at position $\vec z = (z_1, \cdots, z_p) \in \mathbb{R}^p$, distributed according to the distribution $\frac{\mathrm{exp}(\frac12\sum_{\mu=1}^p z_\mu^2)}{(2\pi)^{p/2}} \mathrm{d} z_1 \cdots \mathrm d z_p = \rho(\mathrm d \vec z)$

The RNN potential becomes $h_t(\vec z)$ and evolves according to

$$
\begin{aligned}
\partial_t h(t, \vec z) &= -h(t, \vec z) + \int_{\mathbb{R}^p} w(\vec z, \vec y) \phi(h(t, \vec y)) \rho(\mathrm d \vec y),
\quad
w(\vec z, \vec y) = \sum_{\mu=1}^p \tilde \phi (y_\mu) z_\mu \\
&= -h(t, \vec z) + \sum_{\mu=1}^p z_\mu m_\mu(t)
\end{aligned}
$$

Overlaps are now defined as

$$
m_\mu(t) = \int_{\mathbb R^p} \tilde \phi(y_\mu) \phi(h(t,\vec y)) \rho(\mathrm d \vec y)
$$

Some fixed points :

- $h(\vec z) = z_\mu \quad \mu=1,\cdots,p$
- $h(\vec z) = 0$

## $p$-dimensional closed system

The dynamics of $h(t, \vec z)$ are in a subsystem of dimension $p$ spanned by the ONB of functions $\{e_\mu(\vec z) = z_\mu | \mu=1,\cdots,p\}$, with the scalar product $\langle f, g \rangle = \int_{\mathbb R^p} f(\vec y) g(\vec y) \rho(\mathrm d \vec y)$.

We decompose $h(t, \vec z) = h^\perp(t, \vec z) + \sum_{\mu=1}^p \kappa_\mu(t) z_\mu$, and write the system of equations for $\mu=1,\cdots,p$.

$$
\dot \kappa_\mu(t) = -\kappa_\mu(t) + \int_{\mathbb{R}^p} \tilde\phi(y_\mu) \phi(h(t, \vec y)) \rho(\mathrm d \vec y) = -\kappa_\mu(t) + m_\mu(t)
$$

with initial conditions

$$
\kappa_\mu(0) = \int_{\mathbb{R}^p} y_\mu h(0, \vec y) \rho(\mathrm d \vec y)
$$

and the orthogonal component evolves according to

$$
h^\perp(t, \vec z) = h^\perp(0, \vec z) \mathrm e^{-t},
\quad
h^\perp(0, \vec z) = h(0, \vec z) - \sum_{\mu=1}^p \kappa_\mu(0) z_\mu
$$

## Relation between $p$-dimensional closed system and neural field equation in $\mathbb R^p$

This is consistent with the neural field equation, since

$$
\begin{aligned}
\partial_t h(t, \vec z) &= \partial_t h^\perp(t, \vec z) + \sum_{\mu=1}^p z_\mu \dot \kappa_\mu(t) \\
&= -h^\perp(t, \vec z) - \sum_{\mu=1}^p z_\mu \kappa_\mu(t) + \sum_{\mu=1}^p z_\mu m_\mu(t) \\
&= -h(t, \vec z) + \sum_{\mu=1}^p z_\mu m_\mu(t)
\end{aligned}
$$

# Neural field equation in $[0,1]$

Define a measurable mapping $S : \mathbb{R}^p \rightarrow [0, 1]$.

Let $\mu : [0,1] \rightarrow [0, \infty]$ be a measure on $([0,1], \mathcal B([0,1]))$.

Then we can write

$$
\partial_t h(t, \vec z) = -h(t, \vec z) + \int_{[0,1]} [w(\vec z, \cdot) \phi(h(t, \cdot)) \rho(\cdot)] \circ S^{-1} \; \mathrm d \mu
$$

## Connectivity matrix in $[0,1]$

### General formulation

For finite number of recursive quadrant iterations $n$, we can do a "mean-field approximation" inside each of the $4^n$ segments. Let $\alpha = \{i_1,\cdots,i_{|\alpha|}\}$ be the multi-index corresponding to all neurons of which the embedding in $\mathbb R^p$ gets mapped to the segment $\alpha$ in $[0,1]$. Let $H_\alpha(t) = \frac 1 {|\alpha|} \sum_{i \in \alpha} h_i(t)$ be the (mean) RNN potential of the segment $\alpha$. The connectivity matrix $\tilde J_{\alpha,\beta}$ satisfies

$$
\dot H_\alpha(t) = -H_\alpha(t) + \sum_{\beta \in \text{segments of length } 4^{-n}} \tilde J_{\alpha,\beta} \phi(H_\beta(t))
$$

By substituting the original $h_i(t)$, we find the correct rescaling is given by

$$
\tilde J_{\alpha,\beta} = \frac 1 {|\alpha|} \sum_{i \in \alpha} \sum_{j \in \beta} J_{ij}
$$

### Low-rank case

In the low-rank case, we have

$$
J_{ij} = \frac 1N \sum_{\mu=1}^p F_{\mu,i} G_{\mu,j}
$$

We can define the "mean pattern" inside each bin :

$$
\tilde F_{\mu,\alpha} = \frac{1}{|\alpha|} \sum_{i \in \alpha} F_{\mu,i}, \; \tilde G_{\mu,\alpha} = \frac{1}{|\alpha|} \sum_{i \in \alpha} G_{\mu,i}
$$

Then the connectivity matrix is given by, noting $N = \sum_{\alpha} |\alpha|$,

$$
\tilde J_{\alpha,\beta} = \frac{|\beta|}{\sum_{\beta'} |\beta'|} \left( \sum_{\mu=1}^p \tilde F_{\mu,\alpha} \tilde G_{\mu,\beta} - \delta_{\alpha,\beta} \underbrace{\sum_{\mu=1}^p \sum_{i \in \alpha} \frac{F_{\mu,i}}{|\alpha|} \frac{G_{\mu,i}}{|\alpha|}}_{\gamma_{\alpha}} \right)
$$

Recurrent current is given by

$$
I^\text{rec}_\alpha(t) = \sum_\beta \tilde J_{\alpha,\beta} \phi(H_\beta(t)) = \frac{1}{\sum_{\beta'} |\beta'|} \left( \sum_\beta |\beta| \sum_{\mu=1}^p \tilde F_{\mu,\alpha} \tilde G_{\mu,\beta} \phi(H_\beta(t)) - |\alpha| \tilde \gamma_\alpha \phi(H_\alpha(t)) \right)
$$

The overlaps can be computed as

$$
\tilde m_\mu(t) = \frac{1}{\sum_{\alpha} |\alpha|} \sum_\alpha |\alpha| G_{\mu,\alpha} \phi(H_\alpha(t))
$$