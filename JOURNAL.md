journal < 2023-05-04
--------------------

see also written notebook

TODO : hilbert curve ? converges to something space-filling, but bijective ?
https://en.wikipedia.org/wiki/Hilbert_curve#/media/File:Hilbert_curve_production_rules!.svg
```C
//convert (x,y) to d
int xy2d (int n, int x, int y) {
    int rx, ry, s, d=0;
    for (s=n/2; s>0; s/=2) {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        rot(n, &x, &y, rx, ry);
    }
    return d;
}

//convert d to (x,y)
void d2xy(int n, int d, int *x, int *y) {
    int rx, ry, s, t=d;
    *x = *y = 0;
    for (s=1; s<n; s*=2) {
        rx = 1 & (t/2);
        ry = 1 & (t ^ rx);
        rot(s, x, y, rx, ry);
        *x += s * rx;
        *y += s * ry;
        t /= 4;
    }
}

//rotate/flip a quadrant appropriately
void rot(int n, int *x, int *y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = n-1 - *x;
            *y = n-1 - *y;
        }

        //Swap x and y
        int t  = *x;
        *x = *y;
        *y = t;
    }
}
```




-> repasser sur la 2eme partie de la presentation
1000 bins in 2D <=> 1000 bins in 1D
then do downsampling is important !! (2D we are just discretizing the neural field)
in 1D, it is non-trivial because very discontinuous

we are comparing the neural field in 2D : grid of 4**n neurons, "regularly spaced" (in CDF space)
				 "coarse-grained neuron density/connectivity"
                 with the version in 1D : `2**n` segments. BUT we started from `4**n` segments, then downsampled
				 -> IT STILL WORKS (and better as n -> oo)

change of variable terminology :
	2D : 4**n "coarse-grained densities/sub-populations/" (because we are doing numerical simulations of a neural field)
	1D : 4**n/2**n = 2**n segments        <- the 1D mapping does NOT have an "extra information which then gets downsampled advantage". effectively we are coarse-graining even more
	1D "control" : randomly permute the order of the `4**n` sub-populations, then average down to `2**n`

challenge : finding a mapping from 2d to 1d, which IS bijective, AND resilient to coarse-graining

TODO : how to compare 2D and 1D ? HOW do we downsample (average 2**n) ?
       comparaison convainquante
	   construire une histoire, pour comprendre pourquoi cette comparaison est la bonne





why is this interesting ? -> step back, the question of "dimensionality" of the activity
here low-rank : "linear dimensionality"
in general : decoupled dimensionality
"PCA of activity"

we have a space ("manifold", e.g. Rp, [0,1] etc), place neurons on there
then the activity is a function acting on that space (AND the "dimensionality" of the activity != dim. of the space)
neural field = that function

we can construct examples where the dim of activity (PCA sense) is infinite, BUT the dim of space is finite (1d or 2d)

for nonlinear dim, we need to specify the regularity

we need to consider a couple of regularity/dimensionality

for lab people : if you want to build good models, you need to be able to ask the correct questions
not only "can we write a neural field", but also, "under what constraints ?"





-> dimension fractale, intuition avec nombre et taille discontinuités
"histogramme 2d" pour visualizer [0,1]² -> [0,1]
(plot the 1D embedding, NOT the activity. ceci evite la discussion sur est-ce-que l'état particulier est important)
hypothese : mapping fonctionne ssi dimension fractale > 1

dimension fractale du MAPPING (pas de l'état en particulier)
dans la visualisation 3D
- projection (reshape, diag) -> D = 2
- rec local, z order         -> 2 < D < 3
- far, random                -> D ~= 3
maybe : calculer numériquement dimension fractale des mappings

-> condition nécessaire : mapping bijectif et mesurable dans la limite







talk 2023-05-04
---------------

https://www.overleaf.com/project/6385b2d1c069be7920b70d8d

[1-4] : history
[7-9] : math references. convergence of spatially structured -> neural field
jabin 2022 : theory of graphons. 1/N scaling -> dynamics in 1D embedding (but non-cont. neural field/kernel)
the only thing we know in theory of graphons : the kernel is bounded and integrable. we don't know much about regularity
(here kernel is continuous *in Rp*)
note for me : we can show that if the kernel is bounded and integrable, then the neural field has one unique solution (-> well-defined)


notes on distance in 1D vs distance in 2D
* "close to the line y=sqrt2 x" (but fractals dont get arbitrarily close)
* "size of the discontinuities"
* L1 distance
TODO : taille du bin vs. max span of 2d distances inside the bin
then take integral on [0,1]. we want to show vanishingly few bins with large distances, and so expect ?? -> is this not similar to fractal dimension ?











talk 2023-05-11
---------------

localité :
1. proche in 2D -> proche in 1D (OK projection (obv., x est proche), OK zorder, PAS random)
2. proche in 1D -> proche in 2D (PAS projection, PAS random, OK zorder)
question : 1. and 2. =>? bijection (dans l'idée oui). réciproque ?
"injection dans le deux sens" ?

dimension fractale :
D = 2 (projection)
2 < D < 3 (zorder)
D = 3 (random)
question : 2 < D < 3 =>? bijection ? probably not, because self-intersection (ex : brownian motion has fractal dimension D = 1.5 https://en.wikipedia.org/wiki/Fractional_Brownian_motion#Dimension)
(D permet de distinguer les mappings, mais peut-etre pas un critere tout seul)

bijection = critere suffisant, mais pas forcément nécessaire (on peut imaginer construire le z-mapping, et on repete le 4eme point de chaque quadrant)


3 criteres, ensemble suffisants :
1. mesurable (pour le changement de variables)
2. bijectif dans la limite
3. proche in 1D -> proche in 2D
	* define it ?
	* how do you prove it ? (monotonie (on a la monotonie x > x' -> z > z', pas la réciproque), ou alors compter discontinuities)
	* concept de variation totale
	  (https://en.wikipedia.org/wiki/Bounded_variation#BV_functions_of_several_variables ?)


Questions :
* random mapping is it measurable, does it even converge ?
* far mapping (triv. casse la localite 1D -> 2D). does it even converge, dans le sens (x, y) fixe ---> fixed z as n -> oo
* zmapping a la localite 1D -> 2D (bounded variation)
* critere 1. and 2. =>? BV ? (critere redundant)





talk 2023-05-15
---------------

my findings :
* sum of 1D BV is BV (https://www.whitman.edu/documents/Academics/Mathematics/grady.pdf)
* zmapping can be written as S(x, y) = S1(x) + S2(y)
* S1(x) and S2(y) are both monotone increasing
* monotone increasing functions is BV (https://www.whitman.edu/documents/Academics/Mathematics/grady.pdf)
* problem : generalization to multiple variables.
  we would like that S1(x) is BV in 2D, not just 1D
  basically does S1(x) + const(y) =>? BV(x,y)
  I tried a proof, but could not directly find an equivalence between the 1D def and 2D def (many defs online)
* if we manage to prove S1(x) montone + const(y) => BV, then we have proven zmapping is BV

* wrote an analytical formula for the reshape mapping, we can also write R(x,y) = R1(x) + R2(y)
* problem : both R1(x) and R2(y) are monotone increasing, but we don't expect that R(x,y) is BV ?

* these is a link between BV and fractal dimension (https://www.sciencedirect.com/science/article/pii/S0019357720300161#sec4, https://epub.jku.at/obvulihs/download/pdf/4951527?originalFilename=true sec 2.6), but haven't yet looked into it


=> it looks like reshape and zmapping both are BV in 2D

=> we need a notion of BV from 1D to 2D

Let $f : \mathbb R \rightarrow \mathbb R^2$, let $|| \cdot ||$ be a norm in $\mathbb R^2$ (any norm is OK, since all equivalent (i.e. equal to a constant)). Here we use the $\ell^1$ norm (absolute value).

$$
V(f) = \sup_{P \in \mathcal P} \sum_{i=1}^{n_P} || f(z_{i}) - f(z_{i-1}) ||_1 = \sup_{P \in \mathcal P} \sum_{i=1}^{n_P} | f_1(z_i) - f_1(z_{i-1})| + | f_2(z_i) - f_2(z_{i-1}) |
$$

goal : with this notion of BV, show
* reshape is not BV : show there is a partition for which the sum diverges (as n -> oo)
* zmapping is BV
  * easy start : instead of any z, consider z = 0.b1 b2...bn, n < oo, n_P = 2**(n). take a partition which "takes each discontinuity"

todo (maybe, but not useful) : 
* finish/formalize proof that zmapping is BV (in the 2D -> 1D sense)
* prove monotonic(x) + const(y) => BV(x,y) ? actually in our case this is clear, because we are on a compact set [0,1]², and we can just bound the integral with monotonic(1)
* we are basically certain that zmapping and reshape have BV 2D -> 1D, maybe prove this better later


another idea : "mean error of binning" goes to zero. bins close in 1D should give, on average, bins close in 2D

$$
\frac{1}{4^n}\sum_{i=1}^{4^n}|f_1(i/4^n) - f_1((i-1)/4^n)| + |f_2(i/4^n) - f_2((i-1)/4^n)| \xrightarrow[]{n\to\infty} 0
$$ {#eq-cnt-disc}

i.e. the integral on [0,1] becomes more and more precise, we basically have an "error term" of the integral estimation that goes to zero.

(note : this criteria is probably "too strong" for what we need, i.e. we want the simulation converges to the correct dynamics)

current goal : show this sum (@eq-cnt-disc) is a sufficient condition for the simulation to converge to the correct dynamics (note : here we're not using notions of measurability)

(for later : we don't know if it's actually necessary.)





def pointwise convergent : f_n(x,y) -> z as n -> oo, for all x,y

lemma : the limit of pointwise convergent measure functions is measurable






(!) nomenclature :
* "change of variables" : C1 function, which is invertible
* "co-area formula" : a generalization of the change of vars (compliqué)
* "measure image" (this is what we do) : change the "measure"

(todo : "And even if mathematically such a change
of variables is possible, is it actually feasible to simulate it numerically ?" -> change the "change of vars" to "measure image")

we know : doing the measure image (i.e. having a measurable mapping) guarantees the fact that numerical simulations converge to the analytical solution. (idea : we can approximate L1 functions by piecewise constant function, and this is what we do numerically)

-> and so what we have done so far numerically is valid.
   * zmapping actually works analytically
   * reshape actually doesn't work analytically (these are real analytical results, obtained by numerical simulations. and we know reshape is measurable, because goes to projection, and projections are measurable)
   * random mapping does not even converge pointwise ! (random does not work, because it's not even measurable, so does not even converge to the "analytically correct solution")

(todo : "And even if mathematically such a change
of variables is possible, is it actually feasible to simulate it numerically ?" -> are the resulting dynamics the same ?)



notes 2023-05-16
----------------

* found an upper bound for @eq-cnt-disc applied to Z-mapping, it's V=2 (see ![](pictures/IMG_20230516_111428.jpg))
* started computing the bound for the reshape mapping, stopped midway because i realized the bound vx < 4^n is general : intuitively, a simple permutation cannot vary infinitely.
* i figured it's unlikely that Z-mapping had @eq-cnt-disc -> 0, so did numerical computation
* turns out (see [variation.ipynb](../notebooks/variation.ipynb)) that V (= vx + vy) grows as a power law (straight line on $\log(V) \sim n$, and $\log(V) \lesssim 4^n$)
  => the total discontinuity is basically the same between the different mappings
  ![](pictures/image_20230517.png)
* maybe it's the distribution of discontinuities that matter. Seems like distribution follows power law for Z-mapping and RecLocal, and it much more irregular (non-negligible tail) for Column, Diagonal, Szudzik. ![](pictures/image_20230517_2.png) But computing the mean or median yield comparable results, so no simple statistic seems to discrimiate them
* my current thinking is that that
  * either our definition of locality is not a good one
  * or locality seems to be a "by-product" of the fact that only the mappings that converge to a bijection actually work
    => this makes sense, as a bijection is needed to do the measure image in the integral, so that the equality is verified ! (not sufficient to be measurable)
	=> explore whether bijection is a necessary condition.
	   e.g. a z-mapping but we repeat every second bit of x and y and discard the next (so like 0.b1x b1y b1x b1y b3x b3y b3x b3y ...), so we don't have a bijection (the same point in 2D gets repeated). The reason we repeat and discard is that if we just repeat, we are essentially adding 50% more bits, which we cannot cram into the 2n required bits for z


ideas : 
* study correlations between position in 1D and position in 2D -> actually no because of discontinuities will just jump around
* study whether bijectivity is necessary. can we have a surjection (i.e. position z refers to multiple positions (x,y)) ?
  until now, all our "local" mappings have also been bijective. can we have "local" without bijection ?


notes 2023-05-18
----------------

discussion sur le fait qu'on considere seulement des mappings de type "binary" (ie. S(0.b1x b2x b3x..., 0.b1y b2y b3y ...) -> 0.b1 b2 b3 b4 b5 b6 ..., and x has n bits, y has n bits, z has 2n bits). et que peut-etre que convergent + bijectif dans limite + "binary" => ça marche

maybe the functions that are of "binary" type cannot represent _all_ the measurable mappings, but in essence any thing that we can write numerically is of "binary" form.

might be better to have a result that's weaker but at least we're sure

keep the generalization to any measurable mapping for later.

--- 
compute V pour le column mapping

disc y :
* 4^n - 2^n sauts de taille 2^-n -> (2^n - 1) / 4^n -> 0
* 2^n sauts de taille 1 -> 0

disc x :
* 4^n - 2^n sauts de taille 0 -> 0
* 2^n sauts de taille 2^-n -> 1 / 4^n -> 0

-> this is a problem because we wanted the column mapping to not have V -> 0... (btw this in-line with the numerical graphs, since all the lines are below 4^n, and the gap grows.)

-> our notion of bounded variation is probably not a good one.

note : f2 is ok for finite n, but not defined for infinite n

---

nouvelle definition : 

"variation inside of each bin before averaging"

$$
\frac{1}{2^n}\sum_{i=1}^{2^n} \sup_{\frac{i-1}{2^n}\leq a, b, \leq \frac{i}{2^n}} |f^n_1(4^{-n}\lfloor 4^n a\rfloor) - f^n_1(4^{-n}\lfloor 4^n b\rfloor)| + |f^n_2(4^{-n}\lfloor 4^n a\rfloor) - f^n_2(4^{-n}\lfloor 4^n b\rfloor)| \xrightarrow[]{n\to\infty} 0
$$

equivalent notation : 

$$
\frac{1}{2^n}\sum_{i=1}^{2^n} \sup_{z,z' \in \{\frac{k}{4^n} | k= 1, \cdots, 2^n\}} |f^n_1(\tfrac{i-1}{2^n}+z) - f^n_1(\tfrac{i-1}{2^n}+z')| + |f^n_2(\tfrac{i-1}{2^n}+z) - f^n_2(\tfrac{i-1}{2^n}+z')|
$$

![](pictures/Screenshot_2023-05-19_13-01-40.png)

example $n=2$ : $2^2 = 4$ "big bins", variation inside each along x is $2^{-n}$

todo :
1. simulate the new definition
2. if OK, prove new defintion
3. if OK, how do you accomodate infinite n, i.e. def indep of n

---

topological conjugacy for equivalent dynamics (https://en.wikipedia.org/wiki/Topological_conjugacy)
-> here, the system in 1D is conjugated to the system in  2D (?)

problems here
- we dont have finite timestep
- here our "h" is not homeomorphism


put the accent on the fact we are considering equations : are they conjugated ? we don't care about finite N, since ofc at finite N you can embed in any space (without "avergaing", which is basically doing the integral)


notes 2023-05-19
----------------

TODO : strided binning -> OK, it still works

TODO : ask valentin (1) for a measure theory intro book ; (2) verify that the change of variables is indeed correct ??

---

the new definition works

![](pictures/Screenshot_2023-05-19_13-24-47.png)

I also tried to offset the bins halfway (using `np.roll(F, 2**nrec)`), and it still works

theoretical results for the mean variation :
* column : $V(C) = 1 - 2^{-n} \xrightarrow[]{n\to\infty} 1$
* z-mapping (for even $n$): $V(Z) = 2^{1-\frac n2} - 2^{1-n} \sim 2^{-\frac n2} \xrightarrow[]{n\to\infty} 0$


---

doing some research on measure theory and measurable functions 

![Alt text](pictures/Screenshot_2023-05-19_15-48-41.png)
there is a change of formula for non-bijective functions, but they must be Lipschitz. I don't think the Z-order curve is Lipschitz ?

https://math.stackexchange.com/questions/152338/is-there-a-change-of-variables-formula-for-a-measure-theoretic-integral-that-doe

![Alt text](pictures/Screenshot_2023-05-19_15-51-09.png)
in some sense our discretization is similar to that of the proposition. we have an increasing sequence of functions that converges pointwise to the mappings.

link with pushforward measure : https://en.wikipedia.org/wiki/Pushforward_measure. a map does not need to be invertible to have that the change of variable works ?

[Bogachev V. Measure Theory. Volume I (Springer, 2007)](literature/Bogachev%20V.%20Measure%20Theory.%20Volume%20I%20(Springer,%202007)(514s).pdf) : sections 3.6 and 3.7 talk about change of variables. there is something of concern of measurability of fractal mappings

--- 

a link with dynamical systems ?

$$
\frac{d h_t}{dt} = F(h_t) = -h_t(z) + \int_{[0,1]^2} w_U(z, y) \phi(h_t(y)) dy
$$

Let $dt$ be small.

$$
h_{t + dt} = h_t + dt F(h_t) = f(h_t)
$$

Then for integer $m$ :

$$
h_{t + mdt} = f(f(f(... f(h_t)))) = f^m(h_t)
$$

We can see this as a dynamical system where we repeat $f$


notes 2023-05-30
----------------

Classical neural field papers: \cite{WilCow73} (2d), \cite{Nun74},\cite{Ama77}.

Application of neural field equations \cite{Bre17}.

Idea of cortical column \cite{Mou57,HubWie62}.

Orientation selectivity \cite{BenBar95}.



notes 2023-06-04
----------------

MAYBE :
* implement hilbert curve ? should act the same
* spiral : should be similar to projections. BUT we have locality 1D -> 2D

DISCUSS : 
* show the stability of the fixed points
* show proof for convergence
* mapping p=3 -> [0,1]³ -> [0,1]² -> [0,1]. then we end up with a fractal neural field in [0,1]², which can still be mapped to [0,1] ("is not clearly defined if we don’t specify the regularity (the “smoothness”) of the neural fields along the spatial dimensions").
  in the spirit, the reason this works is because there is still some structure (although no continuity) in [0,1]². this shows that continuity is not a necessary condition, but we only need some "continuity in expectation", "regularity"
  -> yes !
* new abstract
* go through the manuscript to see whether the structure is OK


FURTHER QUESTIONS
* locality might be a side-effect from bijections ? i don't think we can formulate a mappings that is nonlocal, but still bijective. reversely, i think all mapping that are bijective (or can be made bijective by excluding a set of zero measure) also have locality, because in some sense they need the locality to be local.