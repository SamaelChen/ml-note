In this chapter we revisit the classification problem and focus on linear methods for classification. Since our predictor $$G(x)$$ takes values in a discrete set $$\mathcal{G}$$, we can always divide the input space into a collection of regions labeled according to the classification. For an important class of procedures, these *decision boundaries* are linear; this is what we will mean by linear methods for classification.

Actually, all we require is that some monotone transformation of $$\delta_k$$ or $$Pr(G = k|X = x)$$ be linear for the decision boundaries to be linear. For example, if there are two classes, a popular model for the posterior probabilities is
$$
Pr(G=1|X=x)=\frac{exp(\beta_0+\beta^Tx)}{1+exp(\beta_0+\beta^T x)}, \\
Pr(G=2|X=x)=\frac{1}{1+exp(\beta_0+\beta^T x)}.
$$
Here the monotone transformation is the $$logit$$ transformation: $$log[p/(1−p)]$$, and in fact we see that
$$
log \frac{Pr(G=1|X=x)}{Pr(G=2|X=x)} = \beta_0 + \beta^T x.
$$
The decision boundary is the set of points for which the *log-odds* are zero, and this is a hyperplane defined by $$\{ x| \beta_0 + \beta^T x \} = 0$$. We discuss two very popular but different methods that result in linear log-odds or logits: linear discriminant analysis and linear logistic regression.
