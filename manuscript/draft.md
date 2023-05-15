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

## Delayed and rolled RNN

$$
\dot h_i(t) = -h_i(t) + \sum_{j=1}^{N} \frac 1N \sum_{\mu=1}^p \xi_{\mu+1,i} \tilde \phi(\xi_{\mu,j}) \phi(h_j(t - \delta))
$$

## General formulation

$$
J_{ij} = \sum_{\mu=1}^p F_{\mu,i} G_{\mu,j}
$$

# Neural field equation in $\mathbb R^p$

Neurons are at position $\vec z = (z_1, \cdots, z_p) \in \mathbb{R}^p$, distributed according to the distribution $\frac{\mathrm{exp}(\frac12\sum_{\mu=1}^p z_\mu^2)}{(2\pi)^{p/2}} \mathrm{d} z_1 \cdots \mathrm d z_p = \rho(\mathrm d \vec z)$

The RNN potential becomes $h(t, \vec z)$ and evolves according to

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


## Uniform sampling

Defining the change of variables $u_\mu=\mathrm{CDF}(y_\mu), v_\mu=\mathrm{CDF}(z_\mu)$, the neural field becomes

$$
\partial_t h_U(t, \vec v) = -h_U(t, \vec v) + \int_{[0,1]^p} w_U(\vec v, \vec u) \phi(h_U(t, \vec u)) \mathrm d \vec u, \quad
w_U(\vec v, \vec u) = w(\mathrm{CDF}^{-1}(\vec v), \mathrm{CDF}^{-1}(\vec u)), \quad h_U(t, \vec v) = h(t, \mathrm{CDF}^{-1}(\vec v))
$$

We now have a neural field equation on a unit (hyper)cube, and uniform sampling.

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

## Numerical aspects

### Sampling the neural field equation

Numerically, we want to simulate an integral equation. To do this, take a finite number of iterations $n$, and make an integral-to-sum approximation. We sample the $[0,1]^p$ space with $N=4^n$ samples. The reason we sample $[0,1]^p$ is because the CDF takes care of mapping back to $\mathbb R^p$, and we can simply sample uniformly.

$$
\int_{[0,1]^p} w_U(\vec v, \vec u) \phi(h_U(t, \vec u)) \mathrm d \vec u \rightarrow \sum_{j=1}^N w_U(\vec u_i, \vec u_j) \phi(h_U(t, \vec u_j))
$$

$i$ and $j$ correspond to populations, with positions $\vec u_i$ and $\vec u_j$ respectively. These positions are the uniform grid placed in $[0,1]^p$

### Connectivity matrix inside the mapping

We define $\vec z(\alpha) = S(\alpha)$ the 2D point corresponding to the mapping $\alpha$. (details : we need a bounding box to map back). Defining a matrix $Z_{\mu,\alpha} = \vec z(\alpha)_\mu$, we can write down a numerical PDF $\tilde \rho(Z_{1,\alpha},\cdots,Z_{p,\alpha}) = \tilde \rho(Z_{:,\alpha}) = \frac{\rho(Z_{:,\alpha})}{\sum_\beta \rho(Z_{:,\beta})}$ and the following patterns to simulate the embedded $[0,1]$ neural field as a low-rank RNN.

$$
\tilde F_{\mu,\alpha} = Z_{\mu,\alpha}, \quad \tilde G_{\mu,\alpha} = \tilde\phi(Z_{\mu,\alpha}), \quad \tilde J_{\alpha,\beta}=\tilde \rho(Z_{:,\beta}) \sum_{\mu=1}^p \tilde F_{\mu,\alpha} \tilde G_{\mu,\beta}
$$

Doing this is equivalent to the formulation of the binned connectivity matrix.

# Analytical expression of some mappings

We consider $(x,y) \in [0,1]²$. Given $n \in \mathbb N$, we write a (truncated) binary expansion of $x$ as (and similarly for $y$)

$$
b^x_k = \mathrm{Ind}\left\{2^{k-1}x - \lfloor2^{k-1}x\rfloor \geq \frac{1}{2}\right\}, \text{ such that } x= \sum_{k=1}^{n} b^x_k 2^{-k} = 0.b^x_1b^x_2\cdots b^x_{n}
$$

## Z-mapping

Also called Z-order curve, Morton mapping. It conserves locality 2D -> 1D.

$$
z = Z(x,y) = 0.b^x_1 b^y_1 b^x_2 b^y_2 \cdots b^x_n b^y_n = \sum_{k=1}^{n} b^x_k 2^{1-2k} + b^y_k 2^{-2k}
$$

Its inverse is :

$$
\begin{align*}
x = Z^{-1}_x(z) &= 0.b_1 b_3 \cdots b_{2n-1} = \sum_{k=1}^n b_{2k-1} 2^{-k} \\
y = Z^{-1}_y(z) &= 0.b_2 b_4 \cdots b_{2n} = \sum_{k=1}^n b_{2k} 2^{-k}
\end{align*}
$$

## Column mapping

Also called "Reshape mapping" in the code. It converges to a projection on the $x$-axis

$$
z = C(x,y) = 0.b^x_1 b^x_2 \cdots b^x_n b^y_1 b^y_2 \cdots b^y_n
$$

Its inverse is :

$$
\begin{align*}
x = C^{-1}_x(z) &= 0.b_1 b_2 \cdots b_n = \sum_{k=1}^n b_{k} 2^{-k} \\
y = C^{-1}_y(z) &= 0.b_{n+1} b_{n+2} \cdots b_{2n} = \sum_{k=1}^n b_{n+k} 2^{-k}
\end{align*}
$$

## Anti-Z mapping

Also called in code "Far mapping" (initially I didn't see the link with the Z-order curve), because it destroys locality 2D -> 1D.

$$
z = A(x,y) = 0.b^x_n b^y_n b^x_{n-1} b^y_{n-1} \cdots b^x_1 b^y_1 = \sum_{k=1}^{n} b^x_{n+1-k} 2^{1-2k} + b^y_{n+1-k} 2^{-2k}
$$

Its inverse is :

$$
\begin{align*}
x = A^{-1}_x(z) &= 0.b_{2n-1} b_{2n-3} \cdots b_{1} = \sum_{k=1}^n b_{2(n-k)-1} 2^{-k} \\
y = A^{-1}_y(z) &= 0.b_{2n} b_{2n-2} \cdots b_{2} = \sum_{k=1}^n b_{2(n+1-k)} 2^{-k}
\end{align*}
$$