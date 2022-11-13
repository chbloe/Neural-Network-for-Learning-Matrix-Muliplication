#### Strassen Neural Network


An implementation of a neural network aimed at learning Strassen Matrix Multiplication. 

For 2x2 matrices, the network managed to learn the original Strassen Algorithm [1] with 7 multiplications and 18 additions. Furthermore, it figure out an algorithm with 7 multiplications and 16 additions (see below), one addition more than the additionally optimal Strassen-Winograd solution with 15 additions [2].

<br />

2x2 Matrix Multiplication.

\begin{align*} 
A = 
\begin{pmatrix}
A_{1,1} & A_{1,2} \\
A_{2,1} & A_{2,2} \\
\end{pmatrix}
& & B = 
\begin{pmatrix}
B_{1,1} & B_{1,2} \\
B_{2,1} & B_{2,2} \\
\end{pmatrix}
& &AB = 
\begin{pmatrix}
C_{1,1} & C_{1,2} \\
C_{2,1} & C_{2,2} \\
\end{pmatrix} 
\end{align*} 


In matrix element notation, the 7 products are defined as follows,

\begin{align*} 
	I &= -A_{1,2}  (B_{1,2} + B_{2,2}) \\
	II &= A_{2,2} B_{2,1} \\
	III &= ( A_{1,1} - A_{1,2} - A_{2,1} + A_{2,2} ) (B_{2,1} + B_{2,2}) \\
	IV &= A_{2,1} B_{1,1} \\
	V &= (A_{1,1} -A_{2,1} ) (B_{1,1} + B_{1,2} + B_{2,1} + B_{2,2}) \\
	VI &= (A_{1,1} - A_{1,2} - A_{2,1} ) (B_{1,2} + B_{2,1} + B_{2,2} ) \\
	VII &= (A_{1,1} - A_{1,2}) B_{1,2} \\
\end{align*} 


From the terms in these products, we extract the following quantities,

\begin{align*} 
 x_{0} &=  A_{1,1} - A_{2,1}  & y_{0} &= B_{1,2} + B_{2,2} \\
 x_{1} &= A_{1,1} - A_{1,2}  & y_{1} &= B_{2,1} + B_{2,2} \\
 x_{2} &= x_{1} - A_{2,1}   & y_{2} &=  y_{1} +B_{1,2} \\
 x_{3} &= x_{2} + A_{2,2}   & y_{3} &= y_{2} + B_{1,1} \\
\end{align*} 

which require in total 8 additions/subtractions. The 7 products, thus simplify as follows,

\begin{align*} 
I &= -A_{1,2} y_{0}  \\
II &= A_{2,2} B_{2,1} \\
III &= x_{3} y_{1} \\
IV &= A_{2,1} B_{1,1}\\
V &=  x_{0} y_{3}\\
VI &= x_{2} y_{2}  \\
VII &= x_{1} B_{1,2}\\
\end{align*} 

Using these products, we can now construct, analogously as in the original Strassen case [1], the 4 elements of the matrix $AB$. 

\begin{align*}
	C_{1,1} &= I + IV + V - VI \\
	C_{2,1} &= VII - I \\
	C_{1,2} &= II + IV \\
	C_{2,2} &= III - II - VI + VII \\
\end{align*}

The total sum of additions/subtractions is thus 16.



<br />
<br />

###### References

[1] V. Strassen, Gaussian Elimination is not Optimal, NUMER MATH, 13, 354-356 (1969)

[2] S. Winograd, On multiplication of 2 × 2 matrices, LINEAR ALGEBRA APPL, 4, 381-388 (1971)
