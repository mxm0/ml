import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

#read csv
data = np.genfromtxt('traffic_per_hour.csv', dtype=None)
data = [i for i in data if ~np.isnan(i[1])]
data = list(zip(*data))
x = data[0]
y = data[1]

#prepare raw data to plot
plt.scatter(x, y)
plt.xlabel('Days')
plt.ylabel('Hits/Hour')
plt.title('Traffic hits per hour')

#linear fit
polynomial = np.poly1d(np.polyfit(x, y, 1))
ys = polynomial(x)
plt.plot(x, ys, color='red')

#quadratic fit
polynomial = np.poly1d(np.polyfit(x, y, 2))
ys = polynomial(x)
plt.plot(x, ys, color='yellow')

#higher-dimension fit
polynomial = np.poly1d(np.polyfit(x, y, 6))
ys = polynomial(x)
plt.plot(x, ys, color='black')
print('By day: 817 traffic per hour should hit', int(polynomial(817)))

#plt.show()

