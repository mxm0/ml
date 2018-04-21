import numpy as np

# Build a Hilbert matrix of dimension k
k = 5
mx = [[(1.0/(i + j + 1)) for i in range(k)] for j in range(k)]



# a)
#This is the definition of the hilbert martix of dimension k
def hil(k):
    return [[(1.0/(i + j + 1)) for i in range(k)] for j in range(k)]


#this is just some showing the matrices

# Print hilbert matrix.
# np.reshape() split the 2D python array in an actual matrix which numpy can then use.
hm = np.reshape(mx, (k, k))
print(hm)

# Print matrix rank and condition number.
k = 1
mx = [[1.0/(i + j + 1) for i in range(k)] for j in range(k)]
hm = np.reshape(mx, (k, k))
print('Matrix rank and condition number for dimension = 1: ', np.linalg.matrix_rank(hm), ', ', np.linalg.cond(hm))


#b)
#We calculate the needed values for this exercise in each iteration
for k in range(1,31):
    mx = [[1.0/(i + j + 1) for i in range(k)] for j in range(k)]
    hm = np.reshape(mx, (k, k))
    print('Matrix rank and condition number for dimension = 30: ', np.linalg.matrix_rank(hm), ', ', np.linalg.cond(hm))

#c)
# Solving linear equation system for Hilbert matrix of dimension in k_set.
k_set = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]
for k in k_set:
    b = np.array(np.repeat(1, k))
    mx = [[1.0/(i + j + 1) for i in range(k)] for j in range(k)]
    hm = np.reshape(mx, (k, k))
    x = np.linalg.solve(hm, b)
    # Printing the solution, it's really long so I commented it out for now.
    print('Solution: ', x)
    print(np.matmul(hm,x)-b) # care: use matmul


#d)
# The solutions should all be zero because of x is defined as being exactly the value that gets us 0 by this operation. However some entries are not.

#e)
# The reason some entries are not is, that the precision is not enough at some point. The Hilberts matrix condition value gets exponentially higher the
# the higher our k is. So the values needed to calculate the solutions get higher and higher. What follows is that the effect can be seen, that for high k
# more and more values arent 0 any more.