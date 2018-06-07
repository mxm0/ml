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

# Part d -



# Part e -
