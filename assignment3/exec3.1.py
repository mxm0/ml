import numpy as np
import matplotlib.pyplot as plt

# Part a - Random numbers from uniform distribution
n_100 = np.random.uniform(-1, 0, 100)
n_1000 = np.random.uniform(-1, 0, 1000)
n_10000 = np.random.uniform(-1, 0, 10000)
n_100000 = np.random.uniform(-1, 0, 100000)

'''
# Plot raw data
plt.figure('100000')
plt.plot(n_100000, color='green')

plt.figure('10000')
plt.plot(n_10000, color='black')

plt.figure('1000')
plt.plot(n_1000, color='red')

plt.figure('100')
plt.plot(n_100, color='blue')

plt.show()

# Plot histograms
plt.figure('100000')
plt.hist(n_100000, 10)

plt.figure('10000')
plt.hist(n_10000, 10)

plt.figure('1000')
plt.hist(n_1000, 10)

plt.figure('100')
plt.hist(n_100, 10)

plt.show()
'''
'''
It looks like as we get more samples mean, minimum and maximum converge.
'''

# Part b - Random numbers from Gaussian distribution
mu, sigma = 0, 0.1
n_100 = np.random.normal(mu, sigma, 100)
n_1000 = np.random.normal(mu, sigma, 1000)
n_10000 = np.random.normal(mu, sigma, 10000)
n_100000 = np.random.normal(mu, sigma, 100000)

'''
# Plot raw data
plt.figure('100000')
plt.plot(n_100000, color='green')

plt.figure('10000')
plt.plot(n_10000, color='black')

plt.figure('1000')
plt.plot(n_1000, color='red')

plt.figure('100')
plt.plot(n_100, color='blue')

plt.show()

# Plot histograms
plt.figure('100000')
plt.hist(n_100000, 10)

plt.figure('10000')
plt.hist(n_10000, 10)

plt.figure('1000')
plt.hist(n_1000, 10)

plt.figure('100')
plt.hist(n_100, 10)

plt.show()
'''

# Part c - Random numbers from Binomial distribution
n, p = 10, .5
n_100 = np.random.binomial(n, p, 100)
n_1000 = np.random.binomial(n, p, 1000)
n_10000 = np.random.binomial(n, p, 10000)
n_100000 = np.random.binomial(n, p, 100000)
'''
# Plot raw data
plt.figure('100000')
plt.plot(n_100000, color='green')

plt.figure('10000')
plt.plot(n_10000, color='black')

plt.figure('1000')
plt.plot(n_1000, color='red')

plt.figure('100')
plt.plot(n_100, color='blue')

plt.show()

# Plot histograms
plt.figure('100000')
plt.hist(n_100000, 10)

plt.figure('10000')
plt.hist(n_10000, 10)

plt.figure('1000')
plt.hist(n_1000, 10)

plt.figure('100')
plt.hist(n_100, 10)

plt.show()
'''

# Part d - Combining multiple random numbers
 # The values are gaussian or binomial like as expected


for M in [2,3,5,10,20]:
  samples = 200
  m_2 = np.array([sum((np.random.uniform(-1, 0, M))) for i in range(samples)])

  plt.scatter(m_2, range(samples))
  plt.xlabel("Value")
  plt.ylabel("Value Index")
  plt.show()
  plt.hist(m_2)
  plt.show()

#Part e
#We set radius to 10. Then we generate a random angle and then place our point at distance 0 to raidus in this angle
#one might argue that distance 0 is "unfair"

r = 10.0

import math
result = []
samples2 = 20000
for i in range(samples2):
  angle = np.random.uniform(0.0, 2*3.1415926535)
  distance = np.random.uniform(0.0, r)
  result.append([distance * math.cos(angle),distance * math.sin(angle)])



plt.scatter(np.array([result[i-1][0] for i in range(samples2)]), np.array([result[i-1][1] for i in range(samples2)]))
plt.xlabel("X")
plt.ylabel("Y")
plt.show()










































