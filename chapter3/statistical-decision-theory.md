## Statistical Decision Theory

We will develop a small amount of theory that provides a framework for developing models in this section.

We first consider the case of quantitative output. Let's assume $$X \in \mathbb{R}^p$$ and $$Y \in \mathbb{R}$$ with joint distribution $$Pr(X,Y)$$. We seek a function $$f(X)$$ to predict $$Y$$ with input $$X$$. This theory require a _loss function_ $$L(Y,f(x))$$ for penalizing errors in prediction, and by far the most common and convenient is _squares loss error_: $$L(Y,f(x))=(Y-f(x))^2$$. This leads us to a criterion for choosing $$f$$:


$$
\begin{align}
EPE(f) & = E(Y - f(X))^2\\
& = \int [y - f(x)]^2 Pr(dx, dy),
\end{align}
$$


the expected prediction error. Apparently, in most case, we already know $$X$$, but we don't know $$Y$$. To solve this problem, we can change joint probability into conditional probability. That change the equation as:


$$
\begin{align}
EPE(f) &= \int_x \Big\{\int_y [y-f(x)]^2p(y|x)dy\Big\}p(x)dx\\
&= E_XE_{Y|X}([Y - f(x)]^2|X)
\end{align}
$$


and we see that it suffices to minimize EPE point-wise:


$$
f(x) = argmin_c E_{Y|X}([Y-c]^2|X=x).
$$


The solution is


$$
f(x) = E(Y|X=x),
$$


the conditional expectation is also known as the _regression_ function. Thus the best prediction of $$Y$$ at any point $$X = x$$ is the conditional mean, when best is measured by average squared error.

The nearest-neighbor methods attempt to directly implement this recipe using the training data. At each point we have:


$$
\hat{f(x)} = Ave(y_i|x_i \in N_k(x)).
$$


There are two approximations here:

* expectation is approximated by averaging over sample data;
* conditioning at a point is relaxed to conditioning on some region "closed" to target point.

If we get a large training data size $$N$$, as $$N,k \rightarrow \infty$$ and $$k/N \rightarrow 0$$, $$\hat(f(x)) \rightarrow E(Y|X=x)$$.

Now let's consider how to make linear model fit this framework. We just assumed $$f(x) \approx x^T\beta$$. Plugging this function into $$EPE$$ and we can solve for $$\beta$$ theoretically:


$$
\beta = [E(XX^T)]^{-1}E(XY).
$$


The least squares replaces the expectation above by average over the training data.

So both $$k$$-nearest neighbors and least squares end up approximating conditional expectation by averages. But the differences are model assumptions:

* Least squares assumes $$f(x)$$ is well approximated by globally linear model.
* The $$k$$-nearest neighbors assumes $$f(x)$$ is well approximated by a locally constant function.

And if we replace the loss function as $$E|Y-f(x)|$$[^1], the solution will be the conditional median,


$$
\hat{f(x)} = median(Y|X=x),
$$


which is a different measure of location. Meanwhile its estimation are more robust than those for the conditional mean. However, the $$L_1$$ criteria have discontinuities in their derivatives, which has hindered their widespread use.

If our output is categorical, we can use _zero-one_ loss function. The expected predict error is


$$
EPE = E[L(G, \hat{G(X)})],
$$


where again the expectation is taken with respect to the joint distribution $$Pr(G,X)$$. Again we condition, and can write EPE as


$$
EPE = E_X \sum_{k=1}^K L[\mathcal{G_k},\hat{G(X)}]Pr(\mathcal{G_k}|X)
$$


and again it suffices to minimize EPE pointwise:


$$
\hat{G(x)} = argmin_{g \in \mathcal{G}} \sum_{k=1}^K L(\mathcal{G_k}, g)Pr(\mathcal{G}_k|X=x).
$$


With the 0-1 loss function this simplifies to


$$
\hat{G(x)} = argmin_{g \in \mathcal{G}}[1-Pr(g|X=x)]
$$


or simply


$$
\hat{G(X)} = \mathcal{G}_k \ if \ Pr(\mathcal(G)_k|X=x) = \max \limits_{g \in \mathcal{G}}Pr(g|X=x).
$$


This solution is known as _Bayes classifier_. And says that we classify to the most possible class, using the conditional distribution $$Pr(G|X)$$.

[^1]: Call it $$L_1$$ criteria
