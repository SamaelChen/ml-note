# Statistical Learning

## What is Statistical Learning

In essence, statistical learning refers to a set of approaches to estimate $f$. Let's look it in detail.

We can denote the input variables as $X$. The inputs go by different names, like <font face='times new roman'>*predictors, independent variables, features,* or sometimes just <font face='times new roman'>*variables*. The output variable is often called as <font face='times new roman'>*dependent variable* or <font face='times new roman'>*response*, and it typically denoted as $Y$. We use these input variables to predict output variable, and we call this progress as statistical learning.

More generally, suppose we observed a quantitative response $Y$ and $p$ different predictors, $X_1, X_2, \dots, X_p$. We assume there is some relationship between $Y$ and $X=(X_1, X_2, \dots, X_p)$, which can be written in a very general form
$$
Y=f(x)+\epsilon.
$$

Here $f$ is some fixed but unknown function of $X$, and $\epsilon$ is a random <font face='times new roman'>*error term*, which should be independent of $X$ and has mean zero.

### Why Estimate $f$

In many situations, we already have a set of inputs $X$, but the output $Y$ cannot be easily obtained. In this setting, since the error term average zero, we can predict $Y$ using
$$
\hat{Y}=\hat{f}(X),
$$
where $\hat{f}$ represents our estimate for $f$, and $\hat{Y}$ represents the prediction for $Y$.

In general, $\hat{f}$ isn't a perfect estimation of $f$. Then, how to measure the accuracy of our prediction? 
