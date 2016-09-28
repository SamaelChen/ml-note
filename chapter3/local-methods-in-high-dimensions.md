## Local Methods in High Dimensions
As we mentioned above sections, the larger training data size the closer to the theoretical conditional expectations.

However, it will break down in high dimensions, and we call this phenomenon as the *curse of dimensionality*.

First consider the nearest-neighbor procedure for input uniformly distributed in $p$-dimensional unit hypercube.
And we can easily calculate that the range of sample we covered, let's call it $r$, will be the form bellow:
$$
edge^p = r,
$$
and we can find out that $edge=r^{\frac{1}{p}}$. That means if $r$ is constant, when the dimension goes larger, the edge will get closer to 1. This is a very spare data space. If we wanna cover 1% data, the edge we need is 0.63 in 10-dimensions. That makes no sense.

The other consequence of the spare sampling in high dimensions is that all sample points are close to an edge of sample. Consider $N$ data points uniformly distributed in a $p$-dimensional unit ball centered at the origin. The median distance from origin to the closest data point is:
$$
d(p,N)=(1 - \frac{1}{2}^{1/N})^{1/p}.
$$
For $N=500$, $p=10$, $d(p,N) \approx 0.52$, more than half way to boundary. When we are using nearest-neighbor method, we are using the neighbor of some data point to predict it, since the distance is so far, we cannot call it "neighbor".

If the dimension $p=1$, and the data size is $N$, the density is $\frac{1}{N}$. If $p=2$, the density is $\frac{1}{\sqrt{N}}$. In order to keep density equals to $\frac{1}{N}$, the data size we need is $N^2$. In $p$-dimensional space, we need data size $N^p$.

Let's assume we have 1000 training data $x_i$ generated uniformly on $[-1, 1]^p$. Assume the true function is:
$$
Y=f(x)=e^{-8||x||^2},
$$
without any measurement error. We use the 1-nearest-neighbor rule to predict $y_0$ on the test point $x_0=0$. Denote the training set by $\mathcal{T}$. We can compute the expectation error at $x_0$:
$$
\begin{align}
MSE(x_0) &= E_{\mathcal{T}}[f(x_0)-\hat{y_0}]^2 \\
&= E_{\mathcal{T}}[\hat{y}_0-E(\hat{y}_0)+E(\hat{y}_0)-f(x_0)]^2 \\
&= E_{\mathcal{T}}[(\hat{y}_0-E(\hat{y}_0))^2+2(\hat{y}_0-E(\hat{y}_0))(E(\hat{y}_0)-f(x_0))+(E(\hat{y}_0)-f(x_0))^2] \\
&= E_{\mathcal{T}}[\hat{y_0}-E_{\mathcal{T}}(\hat{y_0})]^2+[E_{\mathcal{T}}(\hat{y_0})-f(x_0)]^2 \\
&= Var_{\mathcal{T}}(\hat{y_0})+Bias^2(\hat{y_0}).
\end{align}
$$
Since the expectation of $\hat{y}_0-E(\hat{y}_0)$ is 0, the equation above is feasible.
Unless the nearest neighbor is at 0, $\hat{y_0}$ will be smaller than $f(0)$. When the dimension $p$ grows, the nearest distance will become much more greater, and the estimation tends to be 0 more often.

Suppose, on the other hand, the relation between $Y$ and $X$ is linear,
$$
Y=X^T \beta + \epsilon,
$$
where $\epsilon \sim N(0, \sigma^2)$ and we fit this model by least squares to the training data.

Therefor:
$$
\hat{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T y = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T(\mathbf{X}\beta+\epsilon)= \beta+(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \epsilon
$$

$$
\hat{y}_0 = x_0^T \hat{\beta}=x^T_0\beta + x_0^T(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \epsilon=x_0^T \beta + \sum_{i=1}^N l_i(x_0) \epsilon_i.
$$
Where $l_i(x_0)$ is the $i$th element of $\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}x_0$.

Since the least squares estimate is unbiased, we find that:
$$
\begin{align}
EPE(x_0) &= E_{y_0|x_0} E_{\mathcal{T}}(y_0-\hat{y}_0)^2\\
&= Var(y_0|x_0)+E_{\mathcal{T}}[\hat{y}_0-E_{\mathcal{T}}\hat{y}_0]^2 + [E_{\mathcal{T}}\hat{y}_0 - x^T_0 \beta]^2 \\
&= Var(y_0|x_0) + Var_{\mathcal{T}}(\hat{y}_0) + Bias^2(\hat{y}_0) \\
&= \sigma^2 + E_{\mathcal{T}}x^T_0(\mathbf{X}^T\mathbf{X})^{-1} x_0 \sigma^2 + 0^2
\end{align}
$$

Here we have incurred an additional variance $\sigma^2$ in the prediction error, since our target is not deterministic. There is no bias, and the variance depends on $x_0$. If $N$ is large and $\mathcal{T}$ were selected at random, and assuming $E(X) = 0$, then $\mathbf{X}^T \mathbf{X} \rightarrow N Cov(X)$ and
$$
\begin{align}
E_{x_0}EPEf(x_0) &\sim E_{x_0}x_0^T Cov(X)^{-1} x_0 \sigma^2/N + \sigma^2 \\
&=trace[Cov(X)^−1 Cov(x_0)]\sigma^2/N + \sigma^2 \\
&= \sigma^2(p/N) + \sigma^2.
\end{align}
$$
Here we see that the expected **EPE** increases linearly as a function of $p$, with slope $\sigma^2/N$ . If $N$ is large and/or $\sigma^2$ is small, this growth in variance is negligible (0 in the deterministic case). By imposing some heavy restrictions on the class of models being fitted, we have avoided the curse of dimensionality.
