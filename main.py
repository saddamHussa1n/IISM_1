import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

a = c = 16807
M = 2147483648
K = 64


def seedLCG(initVal):
    global rand
    rand = initVal


def lcg(a, c, m):
    global rand
    rand = (a * rand + c) % m
    return rand / float(m)


def my_rand(n, a, c, m):
    rand_num = np.zeros(n, float)
    for i in range(n):
        rand_num[i] = lcg(a, c, m)
    return rand_num


def MacLaren_Marsaglia(b, c):
    V = b[:K]
    alpha = [0] * 1000
    for t in range(1000):
        s = int(c[t] * K)
        alpha[t] = V[s]
        V[s] = b[t + K]
    return alpha


seedLCG(123)

lcg_result = my_rand(1000, a, c, M)

####################################################################################################

counts, bins = np.histogram(lcg_result, bins=10, range=(0, 1))

stat_0, p_val_0 = stats.kstest(lcg_result, 'norm')
print('Kolmogorov-Smirnov test for LCG\t\t\t\t', p_val_0)

stat_1, p_val_1 = stats.chisquare(counts, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
print('Chi-square test for LCG\t\t\t\t', p_val_1)

####################################################################################################

result_mac = MacLaren_Marsaglia(my_rand(1000 * K, a, c, M), np.random.rand(1000))

counts_1, bins_1 = np.histogram(result_mac, bins=10, range=(0, 1))

stat_2, p_val_2 = stats.kstest(result_mac, 'norm')
print('\nKolmogorov-Smirnov test for MacLaren\t', p_val_2)

stat_3, p_val_3 = stats.chisquare(counts_1, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
print('Chi-square test for MacLaren\t\t\t', p_val_3)

fig, (ax1, ax2) = plt.subplots(
    nrows=1, ncols=2,
    figsize=(16, 8)
)

ax1.scatter(lcg_result[:-1], lcg_result[1:])
ax1.set_title('LCG')

ax2.scatter(result_mac[:-1], result_mac[1:])
ax2.set_title('MacLaren-Marsaglia generator')

plt.show()
