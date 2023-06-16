## A Neural Network Approach for Learning Subcubic Matrix Muliplication Algorithms

A neural network approach is being developed aimed specifically at learning fast i.e. subcubic matrix multiplication. For 2x2 matrices, the network learned the original Strassen Algorithm [1] with 7 multiplications and 18 additions. Additionally, it learned an exact algorithm with 7 multiplications and 16 additions, merely one more addition than the arithmetically optimal Strassen-Winograd algorithm [2].

For a little bit more detail, please have a look at the accompanying [**communication**](https://github.com/christian-loeffeld/Neural-Network-for-Learning-Matrix-Muliplication/blob/main/Exact%20Strassen-Type%20Solutions%20for%202%20x%202%20Matrix%20Multiplication%20from%20Neural%20Network%20Learning.pdf), and for only the raw bits, follow along below.

Just how interesting and promising this approach is, was eventually demonstrated by a team at [**Google Deepmind**](https://www.deepmind.com/) in a Nature publication in 2022, see [here](https://www.nature.com/articles/s41586-022-05172-4).

<br />

### 2x2 Matrix Multiplication.
<br />

<p align="center"><img src="/tex/0ed850caafee4791094897d3dc4c7a49.svg?invert_in_darkmode&sanitize=true" align=middle width=559.6343082pt height=39.452455349999994pt/></p> 


In matrix element notation, the **7 products** are defined as follows,
<br />

<p align="center"><img src="/tex/aa865c965bbc545bbd9952f00f34753b.svg?invert_in_darkmode&sanitize=true" align=middle width=339.646065pt height=164.97714585pt/></p> 


From the terms in these products, we extract the following quantities,
<br />

<p align="center"><img src="/tex/615089bbe1bf373be8ea5e12f6a5e8ce.svg?invert_in_darkmode&sanitize=true" align=middle width=390.69629115pt height=89.9086386pt/></p> 
<br />

which require in total 8 additions/subtractions. The 7 products, thus simplify as follows,
<br />

<p align="center"><img src="/tex/745e6258c8ac18cd556a7c03caf9eb57.svg?invert_in_darkmode&sanitize=true" align=middle width=111.82934895pt height=163.88124059999998pt/></p> 
<br />

Using these products, we can now construct, analogously as in the original Strassen case [1], the 4 elements of the matrix <img src="/tex/5a58df2f9303017b173748509a0aa34c.svg?invert_in_darkmode&sanitize=true" align=middle width=25.622208149999988pt height=22.465723500000017pt/>. 
<br />

<p align="center"><img src="/tex/4b11c713996df2066fab6b02fab3f613.svg?invert_in_darkmode&sanitize=true" align=middle width=206.3829735pt height=89.9086386pt/></p>
<br />

A total sum of **16 additions and subtractions**.


<br />
<br />

#### Essential References

[1] V. Strassen, Gaussian Elimination is not Optimal, NUMER MATH, 13, 354-356 (1969)

[2] S. Winograd, On multiplication of 2 × 2 matrices, LINEAR ALGEBRA APPL, 4, 381-388 (1971)
