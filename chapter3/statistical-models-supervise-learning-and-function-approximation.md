## Statistical Models, Supervise Learning and Function Approximation

Our goal is to find a useful approximation $$\hat{f}(x)$$ to the function $$f(x)$$ that underlies the predictive relationship between the inputs and outputs. In the second section of this chapter, we saw that square error loss lead us to the regression function $$f(x)=E(Y|X=x)$$ for a quantitative response. The class of nearest-neighbor methods can be viewed directly as estimates of this conditional probability. But we've seen that they can fail in at least two ways:
+ if the dimension of input space is high, the nearest neighbors need not be close to the target point, and can result in large error;
+ if special structure is known to exist, this can be used to reduce both of the bias and variance of the estimates.

### A Statistical Model for Joint Distribution $$Pr(X,Y)$$

Suppose in fact that our data arose from a statistical model:
$$
Y=f(x)+\epsilon,
$$
where the random error $$\epsilon$$ has $$E(\epsilon)=0$$ and is independent of $$X$$. Note that for this model, $$f(x)=E(Y|X=x)$$, and in fact the conditional distribution $$Pr(Y|X)$$ depends on **only** through the conditional mean $$f(x)$$

The additive error model is a useful approximation to the truth. It assume we can catch all errors from a deterministic relationship via the error $$\epsilon$$ which includes other unmeasurable variables contribute to $$Y$$, measurement error. Generally, the unmeasurable variables not in input-output pairs $$(X,Y)$$.

The assumption in $$Y=f(x)+\epsilon$$ that the errors are independent and identically distribution is not necessary. With such a model it become natural to use least squares as a data criterion for model estimation. In general, the conditional distribution $$Pr(Y|X)$$ can depend on $$X$$ in complicated ways, but the additive error model precludes these.

Additive error models are typically not used for qualitative response $$G$$. The target function $$p(X)$$ is the conditional density $$Pr(G|X)$$, and this is modeled directly.

### Supervised Learning
Supervise learning attempts to learn $$f$$ by example from a "teacher". The learning algorithm has the property that it can modify the relationship $$\hat{f}$$ between $$X$$ and $$Y$$ in response to differences $$y_i - \hat{f}(x_i)$$ between original and generated output.

### Function Approximation
The goal of function approximation is to obtain a useful approximation to $$f(x)$$ for all $$x$$ in some region of $$\mathbb{R}^p$$, given the representations in $$\mathcal{T}$$.

Many of the approximations we will encounter have associated a set of parameters $$\theta$$ that can be modified to suit the data at hand. For example, the linear model $$f(x)=x^T \beta$$ has $$\theta=\beta$$. Another class of useful approximators can be expressed as *linear basis expansions*
$$
f_{\theta}(x)=\sum_{k=1}^K h_k(x) \theta_k,
$$
where the $$h_k$$ are a suitable set of functions or transformation of the input vector $$x$$. There are lot of examples of $$h_k(x)$$, like polynomial $$x_i^2$$, trigonometric $$cos(x_1)$$, sigmoid transformation common to neural network,
$$
h_k(x)=\frac{1}{1+exp(-x^T \beta_k)}.
$$

We can use least squares to estimate $$\theta$$ in $$f_{\theta}$$ as we did for the linear model, by minimizing the residual sum-of-squares
$$
RSS(\theta)=\sum_{i=1}^N(y_i-f_{\theta}(x_i))^2
$$
as a function of $$\theta$$.

While least squares is generally very convenient, it is not the only criterion used in some cases would not make sense. A more general principle for estimation is *maximum likelihood estimation*.

Suppose we have a random sample $$y_i, i=1, 2, \dots, N$$ from a density $$Pr_{\theta}(y)$$ indexed by some parameters $$\theta$$. The log-probability of the observed sample is
$$
L(\theta)=\sum_{i=1}^N log Pr_{\theta}(y_i).
$$
The principle of maximum likelihood assume that the most reasonable value of $$\theta$$ are those for which the probability of observed sample is largest. Least squares for the additive error model $$Y=f_{\theta}(x)+\epsilon$$, which $$\epsilon \sim N(0, \sigma^2)$$, is equivalent to maximum likelihood using the conditional likelihood
$$
Pr(Y|X, \theta)=N(f_{\theta}(x), \sigma^2).
$$
Although the additional assumption of normality seems more restrictive, the results are the same. The log-likelihood of the data is
$$
L(\theta) = -\frac{N}{2}log(2\pi) - Nlog\sigma - \frac{1}{2\sigma^2} \sum_{i=1}^N(y_i-f_{\theta}(x_i))^2
$$
and the only term involving $$\theta$$ is the last, which is the $$RSS(\theta)$$ up to a scalar negative multiplier.

Suppose we have a model $$Pr(G=\mathcal{G}_k|X=x)=p_{k,\theta}(x), k=1,\dots,K$$ for the conditional probability of each class given $$X$$, indexed by the parameter vector $$\theta$$. Then the log-likelihood(also referred to as cross-entropy) is
$$
L(\theta)=\sum_{i=1}^N \mathrm{log} p_{g_i, \theta}(x_i),
$$
