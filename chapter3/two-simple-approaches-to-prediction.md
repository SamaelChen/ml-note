## Two Simple Approaches to Prediction

There are tow simple but powerful prediction methods in this section, they are the linear model fit by least squares and the $k$-nearest-neighbor prediction rule.

The linear model makes huge assumptions about structures and yields stable but maybe inaccurate prediction. The $k$-nearest-neighbor makes very mild structural assumptions, its predictions often accurate but can be unstable.

### Linear Models and Least Squares

The linear model has maintained the statistics for over 30 years, and remains one of our most important tools.

Given a vector of inputs $X^T = (X_1, X_2, \dots, X_p)$, we predict the output $Y$ via the model
$$
\hat{Y} = \hat{\beta_0} + \sum_{j=1}^p{X_j\hat{\beta_j}}.
$$
The term $\hat{\beta_0}$ is the intercept, also known as the *bias* in machine learning. Often it is convenient to include the constant variable 1 in $X$, include $\hat{\beta_0}$ in the vector of coefficients $\hat{\beta}$, and we can rewrite this model form as an inner product
$$
\hat{Y} = X^T\hat{\beta},
$$
where $X^T$ denotes vector or matrix transpose. Here we are modeling a single output, so $\hat{Y}$ is a scalar; in general $\hat{Y}$ can be a $K$-vector, in which case $\beta$ would be a $p \times K$ matrix of coefficients. In the $(p + 1)$-dimensional input-output space, $(X,\hat{Y})$ represent a hyperplane.

How do we fit the linear model to a set of training dataset? There are many different methods, but by far the most popular is the method of *least squares*. In this approach, we pick the coefficients $\beta$ to minimize the residual sum of squares
$$
RSS(\beta) = \sum_{i=1}^N(y_i-x^T_i\beta)^2.
$$
$RSS(\beta)$ is a quadratic function of the parameters, and hence its minimum always exits, but may not be unique. We can rewrite this formula into matrix notation
$$
RSS(\beta) = (y-X^T\beta)^T(y-X^T\beta),
$$
differentiating w.r.t. $\beta$ we get the normal equations
$$
X^T(y-X\beta)=0.
$$
If $X^TX$ is nonsingular, then the unique solution is given by
$$
\hat{\beta}=(X^TX)^{-1}X^Ty,
$$
and the fitted value at the $i$th input $x_i$ is $\hat{y_i}=\hat{y}(x_i)=x_i^T\hat{\beta}$.

### Nearest-Neighbor Methods
The $k$-nearest neighbor fit for $\hat{Y}$ is defined as follows:
$$
\hat{Y}(x)=\frac{1}{k}\sum_{x_i \in N_k(x)}{y_i},
$$
where $N_k(x)$ is the neighborhood of $x$ defined by the $k$ closest points $x_i$ in the training sample. In words, we find the $k$ observations with $x_i$ closest to $x$ in input space, and average their responses.

### From Least Squares to Nearest Neighbors
The linear decision boundary from least squares is usually smooth and stable to fit, that means it has low variance. But it rely heavily on assumptions, that makes it may contain bias.

On the other hand, the $k$-nearest-neighbor do not rely on the any assumptions, that makes it has low bias. But its decision boundary depends on a handful of training data points, that makes it unstable, and contain high variance.

Each methods has its own situations for which to work best. Linear regression is more suitable for Scenario 1^[Scenario 1: The training data in each class were generated from bivariate Gaussian distributions with uncorrelated components and different means.], and the $k$-nearest-neighbor is more suitable for Scenario 2 ^[Scenario 2: The training data in each class came from a mixture of 10 low-variance Gaussian distributions, with individual means themselves distributed as Gaussian.]

A large subset of the most popular techniques in use today are variants of these two simple procedures. The following list describes some ways in which these simple procedures been enhanced:

+ Kernel method use weights that decrease smoothly to zero with distance from the target points, rather than the effective 0/1 weights used by $k$-nearest-neighbors.
+ In high-dimensional spaces the distance kernels are modified to emphasize some variables rather then others.
+ Local regression fit linear models by locally weighted least squares, rather than fitting constantly locally.
+ Linear models fit a basis expansion of the original inputs allow arbitrarily complex models.
+ Project pursuit and neural network consist of sums of nonlinearly transformed linear models.
