"""
numpy_master_demo.py
 
One-stop NumPy demo program for students.
Covers:
1. Array creation
2. Properties
3. Indexing, slicing, masking
4. Math & vectorization
5. Statistics & aggregations
6. Broadcasting
7. Sorting & searching
8. Logic & boolean ops
9. Reshaping, stacking, splitting
10. Random numbers
11. Linear algebra
12. Calculus-like tools (gradient, diff, trapz)
13. Fourier transforms (FFT)
14. File I/O
15. Matrix construction helpers
16. Advanced math tools
 
Run: python3 numpy_master_demo.py
"""
 
import numpy as np
 
 
def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
 
 
# 1. ARRAY CREATION
def demo_array_creation():
    section("1. ARRAY CREATION")
 
    a = np.array([1, 2, 3, 4])
    print("np.array([1,2,3,4]) ->", a)
 
    b = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print("\n2D array from list of lists:\n", b)
 
    print("\nnp.zeros((2,3)):\n", np.zeros((2, 3)))
    print("\nnp.ones((2,3)):\n", np.ones((2, 3)))
    print("\nnp.full((2,3), 7):\n", np.full((2, 3), 7))
 
    print("\nnp.arange(0, 10, 2):", np.arange(0, 10, 2))
    print("np.linspace(0, 1, 5):", np.linspace(0, 1, 5))
 
    print("\nIdentity matrix np.eye(3):\n", np.eye(3))
 
    print("\nnp.empty((2,2)) (uninitialized):\n", np.empty((2, 2)))
 
 
# 2. PROPERTIES
def demo_properties():
    section("2. ARRAY PROPERTIES")
 
    a = np.array([[1, 2, 3],
                  [4, 5, 6]], dtype=np.int32)
 
    print("Array a:\n", a)
    print("dtype:", a.dtype)
    print("shape:", a.shape)
    print("ndim:", a.ndim)
    print("size:", a.size)
    print("itemsize (bytes per element):", a.itemsize)
    print("nbytes (total bytes):", a.nbytes)
 
 
# 3. INDEXING, SLICING, MASKING
def demo_indexing_slicing_masking():
    section("3. INDEXING, SLICING, MASKING")
 
    a = np.array([10, 20, 30, 40, 50])
    print("a:", a)
    print("a[0]:", a[0])
    print("a[-1] (last):", a[-1])
    print("a[1:4] (slice):", a[1:4])
    print("a[:3]:", a[:3])
    print("a[2:]:", a[2:])
 
    b = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    print("\n2D array b:\n", b)
    print("b[0,0]:", b[0, 0])
    print("b[1,2]:", b[1, 2])
    print("Row 1 -> b[1,:]:", b[1, :])
    print("Column 0 -> b[:,0]:", b[:, 0])
    print("Submatrix b[0:2,1:3]:\n", b[0:2, 1:3])
 
    # Boolean mask
    mask = a > 25
    print("\nBoolean mask a > 25:", mask)
    print("a[mask]:", a[mask])
 
    print("Even elements in b -> b[b % 2 == 0]:", b[b % 2 == 0])
 
    # Fancy indexing
    idx = [0, 2, 4]
    print("\nFancy indexing a[[0,2,4]]:", a[idx])
 
 
# 4. MATH & VECTORIZATION
def demo_math_vectorization():
    section("4. MATH OPERATIONS & VECTORIZATION")
 
    a = np.array([1, 2, 3, 4])
    b = np.array([10, 20, 30, 40])
    print("a:", a)
    print("b:", b)
    print("a + b:", a + b)
    print("a * b:", a * b)
    print("b - a:", b - a)
    print("a ** 2:", a ** 2)
 
    print("\nWith scalars:")
    print("a + 5:", a + 5)
    print("a * 10:", a * 10)
 
    print("\nUnary elementwise functions:")
    print("np.sqrt(a):", np.sqrt(a))
    print("np.square(a):", np.square(a))
    print("np.exp(a):", np.exp(a))
    print("np.log(a):", np.log(a))
 
    angles = np.array([0, np.pi / 2, np.pi])
    print("\nTrigonometry: angles:", angles)
    print("np.sin(angles):", np.sin(angles))
    print("np.cos(angles):", np.cos(angles))
 
 
# 5. STATISTICS & AGGREGATIONS
def demo_statistics_aggregations():
    section("5. STATISTICS & AGGREGATIONS")
 
    data = np.array([10, 20, 20, 40, 50])
    print("data:", data)
    print("mean:", np.mean(data))
    print("median:", np.median(data))
    print("std:", np.std(data))
    print("var:", np.var(data))
    print("min:", np.min(data), "max:", np.max(data))
    print("percentile 25:", np.percentile(data, 25))
    print("percentile 75:", np.percentile(data, 75))
 
    M = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print("\nMatrix M:\n", M)
    print("Overall sum:", M.sum())
    print("Row-wise sum (axis=1):", M.sum(axis=1))
    print("Column-wise sum (axis=0):", M.sum(axis=0))
    print("Row-wise mean:", M.mean(axis=1))
    print("Column-wise mean:", M.mean(axis=0))
    print("Cumulative sum:", np.cumsum(data))
    print("Cumulative product:", np.cumprod(data))
 
 
# 6. BROADCASTING
def demo_broadcasting():
    section("6. BROADCASTING")
 
    M = np.array([[1, 2, 3],
                  [4, 5, 6]])
    v = np.array([10, 20, 30])
    col = np.array([[1],
                    [2]])
 
    print("M:\n", M)
    print("v (1D):", v)
    print("\nM + v (row-wise broadcast):\n", M + v)
 
    print("\ncol:\n", col)
    print("M + col (column-wise broadcast):\n", M + col)
 
    print("\nScalar broadcasting M * 10:\n", M * 10)
 
 
# 7. SORTING & SEARCHING
def demo_sort_search():
    section("7. SORTING & SEARCHING")
 
    a = np.array([50, 10, 40, 20, 30])
    print("Original a:", a)
    print("np.sort(a):", np.sort(a))
    print("np.argsort(a):", np.argsort(a))
 
    # unique
    b = np.array([1, 2, 2, 3, 3, 3])
    print("\nArray b:", b)
    u, counts = np.unique(b, return_counts=True)
    print("unique values:", u)
    print("counts:", counts)
 
    # searchsorted
    sorted_a = np.sort(a)
    print("\nSorted a:", sorted_a)
    pos = np.searchsorted(sorted_a, 25)
    print("Position to insert 25 (searchsorted):", pos)
 
 
# 8. LOGIC & BOOLEAN OPS
def demo_logic_boolean():
    section("8. LOGIC & BOOLEAN OPERATIONS")
 
    a = np.array([True, False, True, True])
    print("a:", a)
    print("np.all(a):", np.all(a))
    print("np.any(a):", np.any(a))
 
    x = np.array([1, 2, 3, 4])
    print("\nx:", x)
    print("x > 2:", x > 2)
    print("np.logical_and(x>1, x<4):", np.logical_and(x > 1, x < 4))
    print("np.logical_or(x==1, x==4):", np.logical_or(x == 1, x == 4))
 
    print("\nnp.isfinite:", np.isfinite([1, np.inf, np.nan]))
    print("np.isnan:", np.isnan([0.0, np.nan, 5.0]))
 
    print("\nnp.where(x>2, 100, 0):", np.where(x > 2, 100, 0))
 
 
# 9. RESHAPING, STACKING, SPLITTING
def demo_reshaping_stacking():
    section("9. RESHAPING, STACKING, SPLITTING")
 
    a = np.arange(12)
    print("a:", a)
 
    b = a.reshape(3, 4)
    print("a.reshape(3,4):\n", b)
 
    print("b.ravel():", b.ravel())
    print("b.flatten():", b.flatten())
    print("b.T (transpose):\n", b.T)
 
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    print("\nx:", x)
    print("y:", y)
    print("np.vstack([x,y]):\n", np.vstack([x, y]))
    print("np.hstack([x,y]):", np.hstack([x, y]))
    print("np.column_stack([x,y]):\n", np.column_stack([x, y]))
 
    c = np.arange(10)
    c1, c2, c3 = np.split(c, [3, 7])
    print("\nSplit c into [0:3],[3:7],[7:]:", c1, c2, c3)
 
 
# 10. RANDOM NUMBERS
def demo_random():
    section("10. RANDOM NUMBERS")
 
    np.random.seed(42)  # reproducible
    print("np.random.rand(2,3):\n", np.random.rand(2, 3))
    print("np.random.randint(0,10,size=(2,5)):\n", np.random.randint(0, 10, size=(2, 5)))
    print("np.random.randn(5):", np.random.randn(5))
 
    choices = np.random.choice([10, 20, 30, 40], size=5)
    print("np.random.choice([10,20,30,40],5):", choices)
 
 
# 11. LINEAR ALGEBRA
def demo_linear_algebra():
    section("11. LINEAR ALGEBRA (np.linalg)")
 
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
 
    print("A:\n", A)
    print("B:\n", B)
    print("A @ B (matrix multiply):\n", A @ B)
 
    print("det(A):", np.linalg.det(A))
    print("inv(A):\n", np.linalg.inv(A))
 
    b_vec = np.array([1, 2])
    x = np.linalg.solve(A, b_vec)
    print("Solve A x = b, b=[1,2], x:", x)
 
    w, v = np.linalg.eig(A)
    print("Eigenvalues of A:", w)
    print("Eigenvectors of A:\n", v)

# 12. CALCULUS-LIKE TOOLS

def demo_calculus_like():

    section("12. CALCULUS-LIKE TOOLS (gradient, diff, trapz)")
 
    x = np.linspace(0, 2 * np.pi, 10)

    y = np.sin(x)

    print("x:", x)

    print("y = sin(x):", y)
 
    # Numerical derivative (gradient)

    dy_dx = np.gradient(y, x)

    print("\nnp.gradient(y, x) (approx dy/dx):", dy_dx)
 
    # Finite difference

    diff_y = np.diff(y)

    print("np.diff(y) (adjacent differences):", diff_y)
 
    # Approximate integral

    integral = np.trapz(y, x)

    print("\nnp.trapz(y, x) (approx integral of sin(x) from 0 to 2π):", integral)
 
 
# 13. FOURIER TRANSFORM

def demo_fft():

    section("13. FOURIER TRANSFORM (np.fft)")
 
    # Simple signal: sum of two sin waves

    t = np.linspace(0, 1, 1000)

    f1, f2 = 5, 20

    signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
 
    fft_vals = np.fft.fft(signal)

    freqs = np.fft.fftfreq(len(t), d=(t[1] - t[0]))
 
    print("Signal length:", len(signal))

    print("FFT first 10 values:", fft_vals[:10])

    print("Freq bins first 10:", freqs[:10])
 
    # Show where max magnitude is (dominant freq)

    magnitudes = np.abs(fft_vals)

    max_idx = np.argmax(magnitudes[1:]) + 1  # skip 0 DC

    print("Dominant frequency approx:", abs(freqs[max_idx]), "Hz")
 
 
# 14. FILE I/O

def demo_file_io():

    section("14. FILE I/O (save/load arrays)")
 
    a = np.arange(10)

    print("Original a:", a)
 
    np.save("numpy_demo_array.npy", a)

    loaded_npy = np.load("numpy_demo_array.npy")

    print("Loaded from numpy_demo_array.npy:", loaded_npy)
 
    np.savetxt("numpy_demo_array.txt", a, fmt="%d")

    loaded_txt = np.loadtxt("numpy_demo_array.txt", dtype=int)

    print("Loaded from numpy_demo_array.txt:", loaded_txt)
 
 
# 15. MATRIX CONSTRUCTION HELPERS

def demo_matrix_helpers():

    section("15. MATRIX CONSTRUCTION HELPERS")
 
    diag_vals = np.array([1, 2, 3])

    D = np.diag(diag_vals)

    print("np.diag([1,2,3]):\n", D)
 
    M = np.arange(1, 10).reshape(3, 3)

    print("\nM:\n", M)

    print("Upper triangle np.triu(M):\n", np.triu(M))

    print("Lower triangle np.tril(M):\n", np.tril(M))
 
    # meshgrid

    x = np.linspace(-1, 1, 3)

    y = np.linspace(-1, 1, 3)

    X, Y = np.meshgrid(x, y)

    print("\nmeshgrid X:\n", X)

    print("meshgrid Y:\n", Y)
 
 
# 16. ADVANCED MATH TOOLS

def demo_advanced_math_tools():

    section("16. ADVANCED MATH TOOLS")
 
    a = np.array([1, 5, 10, 20, 50])

    print("a:", a)

    print("np.clip(a, 5, 20):", np.clip(a, 5, 20))
 
    x = np.array([0, 1, 2, 3, 4])

    y = np.array([1, 2, 1, 2, 1])

    x_new = np.array([0.5, 1.5, 2.5])

    print("\nInterpolation example:")

    print("x:", x)

    print("y:", y)

    print("x_new:", x_new)

    print("np.interp(x_new, x, y):", np.interp(x_new, x, y))
 
    # Polynomial fit: fit y = ax + b over some noisy points

    x2 = np.array([0, 1, 2, 3, 4, 5])

    y2 = np.array([1.1, 2.1, 2.9, 4.2, 4.9, 6.1])

    coeffs = np.polyfit(x2, y2, deg=1)

    print("\nPolynomial fit (degree 1) to points (x2,y2), coeffs:", coeffs)

    print("polyval at x2:", np.polyval(coeffs, x2))
 
    # Correlation and covariance

    d1 = np.array([1, 2, 3, 4, 5])

    d2 = np.array([2, 4, 6, 8, 10])

    print("\nCorrelation matrix between d1 and d2:\n", np.corrcoef(d1, d2))

    print("Covariance matrix between d1 and d2:\n", np.cov(d1, d2))
 
 
def main():

    demo_array_creation()

    demo_properties()

    demo_indexing_slicing_masking()

    demo_math_vectorization()

    demo_statistics_aggregations()

    demo_broadcasting()

    demo_sort_search()

    demo_logic_boolean()

    demo_reshaping_stacking()

    demo_random()

    demo_linear_algebra()

    demo_calculus_like()

    demo_fft()

    demo_file_io()

    demo_matrix_helpers()

    demo_advanced_math_tools()
 
 
if __name__ == "__main__":

    main()

 
