# Part a - Different types of sparse matrix and csr memoru layout.

'''
There are different types of sparse matrices because each type provides benefits when either storing or accessing the original sparse matrix data. The csr_matrix stores the sparse matrix as 3 one dimensional arrays. Each array mantains different information about the data. The first array contains all the non-zero values in row-major order. The second array counts the number of non-zero values as they grow for each row, starting from zero. Finally the last array contains the column index of the first array values. 
'''

# Part b - Most efficient actions for each type of spare matrix.

'''
DOK, LIL or COO are the most efficient if we want to add non-zero values to the matrix. While CSR, CSC or BSR are better for dot products and multiplication. They are efficient because the data structure used to mantain the data is optimized to perform those actions. For instance LIL uses a list for each row, often sorted, for faster lookup. While as we have seen in CSR we mantain more info about the data which makes it easier to perform calculations but more tedious to modify.
'''

# Part c - Compute sparsity of matrix

'''
I can't find a specific function but we can ask the matrix to return the number of non-zero values. Substract the number of non-zero values from the size of the matrix and divide everything for the size of the matrix.
'''

# Part d - Build tridiagonal matrix

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse import linalg
import timeit
import tracemalloc

N = 100000

A = diags([1, -2, 1], [-1, 0, 1], shape=(N, N))
#print(A.todense())

# Part e - Compare linalg memory usage and execution time

b = np.ones(N)

# Printing exec. time
print("N = 100")
print("CSR Matrix time:",timeit.timeit("linalg.spsolve(csr_matrix(A), b)", setup='import numpy as np;from scipy.sparse import diags;from scipy.sparse import linalg;from scipy.sparse import csr_matrix;N = 100;A = diags([1, -2, 1], [-1, 0, 1], shape=(N, N));b = np.ones(N)', number=1))
print("Normal matrix time:", timeit.timeit("linalg.spsolve(A.todense(), b)", setup='import numpy as np;from scipy.sparse import diags;from scipy.sparse import linalg;N = 100;A = diags([1, -2, 1], [-1, 0, 1], shape=(N, N));b = np.ones(N)', number=1))

print("\nN = 1000")
print("CSR Matrix time:",timeit.timeit("linalg.spsolve(csr_matrix(A), b)", setup='import numpy as np;from scipy.sparse import diags;from scipy.sparse import linalg;from scipy.sparse import csr_matrix;N = 1000;A = diags([1, -2, 1], [-1, 0, 1], shape=(N, N));b = np.ones(N)', number=1))
print("Normal matrix time:", timeit.timeit("linalg.spsolve(A.todense(), b)", setup='import numpy as np;from scipy.sparse import diags;from scipy.sparse import linalg;N = 1000;A = diags([1, -2, 1], [-1, 0, 1], shape=(N, N));b = np.ones(N)', number=1))

print("\nN = 10000")
print("CSR Matrix time:",timeit.timeit("linalg.spsolve(csr_matrix(A), b)", setup='import numpy as np;from scipy.sparse import diags;from scipy.sparse import linalg;from scipy.sparse import csr_matrix;N = 10000;A = diags([1, -2, 1], [-1, 0, 1], shape=(N, N));b = np.ones(N)', number=1))
print("Normal matrix time:", timeit.timeit("linalg.spsolve(A.todense(), b)", setup='import numpy as np;from scipy.sparse import diags;from scipy.sparse import linalg;N = 10000;A = diags([1, -2, 1], [-1, 0, 1], shape=(N, N));b = np.ones(N)', number=1))

print("\nN = 100000")
print("CSR Matrix time:",timeit.timeit("linalg.spsolve(csr_matrix(A), b)", setup='import numpy as np;from scipy.sparse import diags;from scipy.sparse import linalg;from scipy.sparse import csr_matrix;N = 100000;A = diags([1, -2, 1], [-1, 0, 1], shape=(N, N));b = np.ones(N)', number=1))

# Printing memory usage
import tracemalloc
tracemalloc.start(25)

linalg.spsolve(A.todense(), b)

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('traceback')

# pick the biggest memory block
stat = top_stats[0]
print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
for line in stat.traceback.format():
    print(line)

'''
The largest N for the dense matrix is 10000.
'''
