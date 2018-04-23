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
m_2 = np.random(sum(np.random.uniform(2)), 10000)
m_3 = np.random(sum(np.random.uniform(3)), 10000)
m_5 = np.random(sum(np.random.uniform(5)), 10000)
m_10 = np.random(sum(np.random.uniform(10)), 10000)
m_20 = np.random(sum(np.random.uniform(20)), 10000)

