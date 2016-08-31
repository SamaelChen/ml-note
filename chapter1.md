# Overview

## Notation

+ We will use $$n$$ to represent the number of distinct data points, or observations.

+ Let $$p$$ denote the number of variables that are available for use in making predictions.

+ We will let $$x_{ij}$$ represent the value of the $$j$$th variable for the $$i$$th observation, where $$i=1,\ 2,\ \dots,\ n$$, $$j=1,\ 2,\ \dots,\ p$$.

+ We let $$\textbf{X}$$ denote a $$n \times p$$ matrix whose $$(i,j)$$th element is $$x_{ij}$$.That is,
$$
\textbf{X}=\begin{pmatrix}
x_{11} & x_{12} & \dots & x_{1p} \\
x_{21} & x_{22} & \dots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \dots & x_{np}
\end{pmatrix}
$$

+ We denote $$x_i$$ as the $$i$$th row of  $$\textbf{X}$$. $$x_i$$ is a vector of length $$p$$, containing the $$p$$ variable measurements for the $$i$$th observation. That is,
$$
x_i=\begin{pmatrix}
x_{i1} \\
x_{i2} \\
\vdots \\
x_{ip}
\end{pmatrix}
$$

,+ If we are interested in the columns of $$\textbf{X}$$, which we will write as $$\textbf{x}_1,\ \textbf{x}_2,\ \dots,\ \textbf{x}_p $$. Each is a vector of length $$n$$ That is,

$$
\textbf{x}_j=\begin{pmatrix}
\textbf{x}_{1j} \\
\textbf{x}_{2j} \\
\vdots \\
\textbf{x}_{nj}
\end{pmatrix}
$$

+ If we use these notations, we can write $$\textbf{X}$$ as
$$
\textbf{X}=\begin{pmatrix}
\textbf{x}_1 & \textbf{x}_2 & \dots & \textbf{x}_p
\end{pmatrix}
$$

$$
\textbf{X}=\begin{pmatrix}
x^T_1 \\
x^T_2 \\
\vdots \\
x^T_n
\end{pmatrix}
$$

+ The $$^T$$ notation denotes the *transpose* of a matrix or vector. For example,
$$
\textbf{X}^T=\begin{pmatrix}
x_{11} & x_{21} & \dots & x_{n1} \\
x_{12} & x_{22} & \dots & x_{n2} \\
\vdots & \vdots & \ddots & vdots \\
x_{1p} & x_{2p} & \dots & x_{np}
\end{pmatrix}
$$
while
$$
x^T_i=\begin{pmatrix}
x_{i1} & x_{i2} & \dots & x_{ip}
\end{pmatrix}
$$

+ We use $$y_i$$ to denote the $$i$$th observation of the variable on which we wish to predict. Hence, we write the set of all $$n$$ observations in vector form as
$$
\textbf{y}=\begin{pmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{pmatrix}
$$

+ We always denote a vector of length n in *<font face='Computer Modern'>lower case bold* e.g.
$$
\textbf{a}=\begin{pmatrix}
a_1 \\
a_2 \\
\vdots \\
a_n
\end{pmatrix}
$$

+ If a vector not of length n will be denoted in *<font face='Computer Modern'> lower case normal font*, e.g. $$a$$.

+ Matrix will be denoted using *<font face='Computer Modern'> bold capitals*, such as $$\textbf{A}$$.

+ Random variables will be denoted using *<font face='Computer Modern'> capital normal font*, e.g. $$A$$, regardless of their dimensions.

+ To indicate that an object is a scalar, we will use the notation $$a \in \mathbb{R} $$. To indicate that it is a vector of length $$k$$, we will use $$a \in \mathbb{R}^k$$. We will indicate that an object is a $$r \times s$$ matrix using $$\textbf{A} \in \mathbb{R}^{r \times s}$$.

+ Suppose that $$ \textbf{A} \in \mathbb{R}^{r \times d} $$ and $$ \textbf{B} \in \mathbb{R}^{d \times s} $$. Then the product of $$\textbf{A}$$ and $$\textbf{B}$$ is denoted $$\textbf{AB}$$. That is, $$ (\textbf{A}\textbf{B})_{ij}=\begin{matrix} \sum_{k=1}^d a_{ik}b_{kj} \end{matrix}$$. As an example, consider
$$
\textbf{A}=\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix} \  and \
\textbf{B}=\begin{pmatrix}
5 & 6 \\
7 & 8
\end{pmatrix}
$$
Then
$$
\textbf{AB}=\begin{pmatrix}1 & 2 \\ 3 & 4 \end{pmatrix}\begin{pmatrix}5 & 6 \\ 7 & 8 \end{pmatrix}=\begin{pmatrix}1\times5+2\times7 & 1\times6+2\times8 \\ 3\times5+4\times7 & 3\times6+4\times8 \end{pmatrix} = \begin{pmatrix}19 & 22 \\ 43 & 50 \end{pmatrix}
$$
