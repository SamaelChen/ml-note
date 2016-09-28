## Matrix Operation
+ Denote $$tr(A)=\sum_{i=1}^n a_{ii}$$
+ $$tr(\mathbf{AB})=tr(\mathbf{BA})$$
    + prove: $$\begin{align}tr(\mathbf{AB})&=\sum_{i=1}^n(\mathbf{AB})_{ii}\\
    &= \sum_{i=1}^n \sum_{j=1}^m \mathbf{A}_{ij}\mathbf{B}_{ji} \\
    &= \sum_{j=1}^m \sum_{i=1}^n \mathbf{B}_{ji}\mathbf{A}_{ij} \\
    &= \sum_{j=1}^m(\mathbf{BA})_{jj} \\
    &= tr(\mathbf{BA})
    \end{align}$$

+ $$tr(\mathbf{ABC}) = tr(\mathbf{CAB}) = tr(\mathbf{BCA})$$
+ $$\frac{\partial{tr(\mathbf{AB})}}{\partial{\mathbf{A}}}=\frac{\partial{tr(\mathbf{BA})}}{\partial{\mathbf{A}}}=\mathbf{B}^T$$
    + prove:
    $$
    tr(\mathbf{AB}) = \sum_{i=1}^m \sum_{j=1}^n a_{ij}b_{ji} \\
    \because \frac{\partial{tr(\mathbf{AB})}}{\partial{a_{ij}}}=b_{ji} \\
    \therefore \frac{\partial{tr(\mathbf{AB})}}{\partial{\mathbf{A}}}=\mathbf{B}^T
    $$

+ $$\frac{\partial{tr(\mathbf{A}^T\mathbf{B})}}{\partial{\mathbf{A}}}=\frac{\partial{tr(\mathbf{B} \mathbf{A}^T)}}{\partial{\mathbf{A}}}=\mathbf{B}$$
+ $$tr(\mathbf{A})=tr(\mathbf{A}^T)$$
+ If $$a \in \mathbb{R}, tr(a) = a$$
+ $$
\begin{align}
\frac{\partial{tr(\mathbf{AB}\mathbf{A}^T\mathbf{C})}}{\partial{\mathbf{A}}} &= \frac{\partial{tr(\mathbf{AB}\mathbf{A}^T\mathbf{C})}}{\partial{\mathbf{A}}}+\frac{\partial{tr(\mathbf{A}^T\mathbf{C}\mathbf{AB})}}{\partial{\mathbf{A}}} \\
&= (\mathbf{B}\mathbf{A}^T\mathbf{C})^T + \mathbf{CAB} \\
&= \mathbf{C}^T\mathbf{A}\mathbf{B}^T + \mathbf{CAB}
\end{align}
$$
