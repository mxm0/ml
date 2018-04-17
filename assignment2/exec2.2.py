import numpy as np

# Build a Hilbert matrix of dimension k
k = 5
mx = [[1/(i + j + 1) for i in range(k)] for j in range(k)]

# Print hilbert matrix.
# np.reshape() split the 2D python array in an actual matrix which numpy can then use.
hm = np.reshape(mx, (k, k))
print(hm)

# Print matrix rank and condition number.
k = 1
mx = [[1/(i + j + 1) for i in range(k)] for j in range(k)]
hm = np.reshape(mx, (k, k))
print('Matrix rank and condition number for dimension = 1: ', np.linalg.matrix_rank(hm), ', ', np.linalg.cond(hm))

k = 30
mx = [[1/(i + j + 1) for i in range(k)] for j in range(k)]
hm = np.reshape(mx, (k, k))
print('Matrix rank and condition number for dimension = 30: ', np.linalg.matrix_rank(hm), ', ', np.linalg.cond(hm))

# Solving linear equation system for Hilbert matrix of dimension in k_set.
k_set = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]
for k in k_set:
    b = np.array(np.repeat(1, k))
    mx = [[1/(i + j + 1) for i in range(k)] for j in range(k)]
    hm = np.reshape(mx, (k, k))
    x = np.linalg.solve(hm, b)
    # Printing the solution, it's really long so I commented it out for now.
    #print('Solution: ', x)
