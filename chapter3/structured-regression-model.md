## Structured Regression Models
Consider the $$RSS$$ criterion for an arbitrary function $$f$$,
$$
RSS(f)=\sum_{i=1}^N(y_i-f(x_i))^2.
$$
Minimizing may lead to infinitely many solutions: any $$\hat{f}$$ passing through the training points $$(x_i, y_i)$$ is a solution.

In order to obtain useful results for finite N , we must restrict the eligible solutions to $$RSS$$ to a smaller set of functions. How to decide on the nature of the restrictions is based on considerations outside of the data. These restrictions are sometimes encoded via the parametric representation of $$f_{\theta}$$, or may be built into the learning method itself, either implicitly or explicitly. These restricted classes of solutions are the major topic of this book.

In general the constraints imposed by most learning methods can be described as complexity restrictions of one kind or another. This usually means some kind of regular behavior in small neighborhoods of the input space. That is, for all input points x sufficiently close to each other in some metric, $$\hat{f}$$ exhibits some special structure such as nearly constant, linear or low-order polynomial behavior. The estimator is then obtained by averaging or polynomial fitting in that neighborhood.

One fact should be clear by now. Any method that attempts to produce locally varying functions in small isotropic neighborhoods will run into problems in high dimensions—again the curse of dimensionality. And conversely, all methods that overcome the dimensionality problems have an associated—and often implicit or adaptive—metric for measuring neighborhoods, which basically does not allow the neighborhood to be simultaneously small in all directions.
