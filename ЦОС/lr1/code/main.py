import numpy as np
import matplotlib.pyplot as plt
import math


def get_var():
    variables = {'Nb': 11}
    variables['N'] = 30 + variables['Nb'] % 5
    variables['T'] = 0.0005 * (1 + variables['Nb'] % 3)
    variables['a'] = (-1) ** variables['Nb'] * (0.8 + 0.005 * variables['Nb'])
    variables['C'] = 1 + variables['Nb'] % 5
    variables['w0'] = math.pi / (6 + variables['Nb'] % 5)
    variables['m'] = 5 + variables['Nb'] % 5
    variables['U'] = variables['Nb']
    variables['n0'] = variables['Nb'] % 5 + 3
    variables['n_imp'] = variables['Nb'] % 5 + 5
    variables['B1'] = 1.5 + variables['Nb'] % 5
    variables['B2'] = 5.7 - variables['Nb'] % 5
    variables['B3'] = 2.2 + variables['Nb'] % 5
    variables['w1'] = math.pi / (4 + variables['Nb'] % 5)
    variables['w2'] = math.pi / (8 + variables['Nb'] % 5)
    variables['w3'] = math.pi / (16 + variables['Nb'] % 5)
    variables['a1'] = 1.5 - variables['Nb'] % 5
    variables['a2'] = 0.7 + variables['Nb'] % 5
    variables['a3'] = 1.4 + variables['Nb'] % 5
    variables['x1'] = lambda k: variables['B1'] * np.sin(variables['w1'] * k)
    variables['x2'] = lambda k: variables['B2'] * np.sin(variables['w2'] * k)
    variables['x3'] = lambda k: variables['B3'] * np.sin(variables['w3'] * k)
    return variables


vv = get_var()
print(vv)
x = np.linspace(0, (vv['N'] - 1) * vv['T'])
x_norm = np.linspace(0, vv['N'] - 1, vv['N'])


def dirak(_x, m=0):
    y = np.zeros(_x.shape)
    y[_x == m] = 1
    return y


def exp(_x, m=0, part='real'):
    y = np.zeros(_x.shape)
    ans = np.float_power(vv['a'] + 0j, _x[_x >= m] - m)
    y[_x >= m] = ans.real if part == 'real' else ans.imag
    return y


def exp2(_x, part='real'):
    ans = vv['C'] * np.exp(1j * vv['w0'] * _x)
    return ans.real if part == 'real' else ans.imag


def hs(_x, m=0):
    return np.heaviside(_x - m, 1)


def rect(_x, m=vv['n0']):
    y = np.zeros(_x.shape)
    y[(m <= _x) & (_x <= m + vv['n_imp'] - 1)] = vv['U']
    return y


# 1
#plt.stem(x, dirak(x, 0), use_line_collection=True)
#plt.show()
#plt.stem(x_norm, dirak(x_norm, 0), use_line_collection=True)
#plt.show()


#2
#plt.stem(x, hs(x), use_line_collection=True)
#plt.show()
#plt.stem(x_norm, hs(x_norm), use_line_collection=True)
#plt.show()


#3
#plt.stem(x, exp(x, 0), use_line_collection=True)
#plt.show()
#plt.stem(x_norm, exp(x_norm, 0), use_line_collection=True)
#plt.show()
#plt.stem(x, exp(x, 0, 'imag'), use_line_collection=True)
#plt.show()
#plt.stem(x_norm, exp(x_norm, 0, 'imag'), use_line_collection=True)
#plt.show()


#4
#plt.stem(x, exp2(x), use_line_collection=True)
#plt.show()
#plt.stem(x_norm, exp2(x_norm), use_line_collection=True)
#plt.show()
#plt.stem(x, exp2(x, 'imag'), use_line_collection=True)
#plt.show()
#plt.stem(x_norm, exp2(x_norm, 'imag'), use_line_collection=True)
#plt.show()


#5
#plt.stem(x_norm, dirak(x_norm, vv['m']), use_line_collection=True)
#plt.show()
#plt.stem(x_norm, hs(x_norm, vv['m']), use_line_collection=True)
#plt.show()
#plt.stem(x_norm, exp(x_norm, vv['m'], 'real'), use_line_collection=True)
#plt.show()
#plt.stem(x_norm, exp(x_norm, vv['m'], 'imag'), use_line_collection=True)
#plt.show()


#6
#plt.stem(x_norm, rect(x_norm), use_line_collection=True)
#plt.show()

#7
#x_7 = np.linspace(0, 5 * vv['N'] - 1, 5 * vv['N'])
#ans = vv['a1'] * vv['x1'](x_7) + vv['a2'] * vv['x2'](x_7) + vv['a3'] * vv['x3'](x_7)
#plt.stem(x_7, ans, use_line_collection=True)
#plt.show()
#plt.stem(x_7, vv['x1'](x_7), use_line_collection=True)
#plt.show()
#plt.stem(x_7, vv['x2'](x_7), use_line_collection=True)
#plt.show()
#plt.stem(x_7, vv['x3'](x_7), use_line_collection=True)
#plt.show()
#print('Среднее - {}, Энергия - {}, Мощность - {}'.format(ans.mean(), np.power(ans, 2).sum(), np.power(ans, 2).mean()))

#8
#def s5(_x):
#    return (np.abs(vv['a']) ** _x) * np.cos(vv['w0'] * _x)
#plt.stem(x_norm, s5(x_norm), use_line_collection=True)
#plt.show()


#9
#x_9 = np.linspace(0, 3 * vv['N'] - 1, 3 * vv['N'])
#y = np.zeros(shape=x_9.shape)
#for i in range(5):
#    y += rect(x_9, 2 * vv['n_imp'] * i)
#plt.stem(x_9, y, use_line_collection=True)
#plt.show()
