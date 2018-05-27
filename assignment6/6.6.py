
#a)

It depends on the implementation. I think it would be possible to implement an algorithm that would have a complexity of d*n while k is much smaller than n.
But with an naive implementation the complexity should be O(d*n*k)

#b)

The complexity should be n*d*m

#c)
In each of these cases the data point should be the bigger values:
n*k
But in the case of SVM there are the gradients (m*d) and in the case of the nearest neighbor,there is no additional information.

#d)
Space: 256*1000 = 256000
Runtime: 10000*256*1000 = 2_560_000_000