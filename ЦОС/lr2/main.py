import numpy as np
import matplotlib.pyplot as plt
import math

# task1
"""
h_3 = lambda x: np.sin(3*math.pi*x)/(3*np.sin(x*math.pi))
h_5 = lambda x: np.sin(5*math.pi*x)/(5*np.sin(x*math.pi))
h_7 = lambda x: np.sin(7*math.pi*x)/(7*np.sin(x*math.pi))
h_9 = lambda x: np.sin(9*math.pi*x)/(9*np.sin(x*math.pi))

t = np.linspace(0.001, 1.0, 64)

plt.plot(t, h_3(t), label=r'$H_3$')
plt.plot(t, h_5(t), label=r'$H_5$')
plt.plot(t, h_7(t), label=r'$H_7$')
plt.plot(t, h_9(t), label=r'$H_9$')
plt.xlabel('t')
plt.ylabel(r'$H(t)$')
plt.legend()
plt.show()
"""

# task2
"""
h_7 = lambda x: 1 / 21 * (7 + 12 * np.cos(x) + 6 * np.cos(2 * x) - 4 * np.cos(3 * x))
h_9 = lambda x: 1 / 231 * (59 + 108 * np.cos(x) + 78 * np.cos(2 * x) + 28 * np.cos(3 * x) - 42 * np.cos(4 * x))
h_11 = lambda x: 1 / 429 * (
            89 + 168 * np.cos(x) + 138 * np.cos(2 * x) + 88 * np.cos(3 * x) + 18 * np.cos(4 * x) - 72 * np.cos(
        5 * x))
h_13 = lambda x: 1 / 143 * (
            25 + 48 * np.cos(x) + 42 * np.cos(2 * x) + 32 * np.cos(3 * x) + 18 * np.cos(4 * x) - 22 * np.cos(6 * x))

t = np.linspace(0, 2 * math.pi, 64)

plt.plot(t, h_7(t), label=r'$H_7$')
plt.plot(t, h_9(t), label=r'$H_9$')
plt.plot(t, h_11(t), label=r'$H_{11}$')
plt.plot(t, h_13(t), label=r'$H_{13}$')
plt.xlabel('t')
plt.ylabel(r'$H(t)$')
plt.legend()
plt.show()
"""

# task3
"""
h_9 = lambda x: 1 / 429 * (179 + 270 * np.cos(x) + 60 * np.cos(2 * x) - 110 * np.cos(3 * x) + 30 * np.cos(4 * x))
h_11 = lambda x: 1 / 429 * (143 + 240 * np.cos(x) + 120 * np.cos(2 * x) - 20 * np.cos(3 * x) - 90 * np.cos(4 * x) + 36 * np.cos(5 * x))
h_13 = lambda x: 1 / 2431 * (677 + 1200 * np.cos(x) + 780 * np.cos(2 * x) + 220 * np.cos(3 * x) - 270 * np.cos(4 * x) - 396 * np.cos(5 * x) + 220 * np.cos(6 * x))
h_15 = lambda x: 1 / 46189 * (11063 + 20250 * np.cos(x) + 15000 * np.cos(2 * x) + 7510 * np.cos(3 * x) - 330 * np.cos(4 * x) - 5874 * np.cos(5 * x) - 5720 * np.cos(6 * x) + 4290 * np.cos(7 * x))

t = np.linspace(0, 2 * math.pi, 64)

plt.plot(t, h_9(t), label=r'$H_9$')
plt.plot(t, h_11(t), label=r'$H_11$')
plt.plot(t, h_13(t), label=r'$H_{13}$')
plt.plot(t, h_15(t), label=r'$H_{15}$')
plt.xlabel('t')
plt.ylabel(r'$H(t)$')
plt.legend()
plt.show()
"""


# task4
"""
h_21 = lambda x: 1/320 * (74 + 134*np.cos(2*math.pi*x) + 92*np.cos(4*math.pi*x) + 42*np.cos(6*math.pi*x) + 6*np.cos(8*math.pi*x) - 10*np.cos(10*math.pi*x) - 12*np.cos(12*math.pi*x) - 6*np.cos(14*math.pi*x))
h_15 = lambda x: 1/350 * (60 + 114*np.cos(2*math.pi*x) + 94*np.cos(4*math.pi*x) + 66*np.cos(6*math.pi*x) + 36*np.cos(8*math.pi*x) + 12*np.cos(10*math.pi*x) - 4*np.cos(12*math.pi*x) - 10*np.cos(14*math.pi*x) - 10*np.cos(16*math.pi*x) - 6*np.cos(18*math.pi*x) - 2*np.cos(20*math.pi*x))

t = np.linspace(0, 1, 64)

plt.plot(t, h_21(t), label=r'$H_21$')
plt.plot(t, h_15(t), label=r'$H_15$')
plt.xlabel('t')
plt.ylabel(r'$H(t)$')
plt.legend()
plt.show()
"""

#task5
"""
def task1_inner():
    h_3 = lambda x: np.sin(3 * math.pi * x) / (3 * np.sin(x * math.pi))
    h_5 = lambda x: np.sin(5 * math.pi * x) / (5 * np.sin(x * math.pi))
    h_7 = lambda x: np.sin(7 * math.pi * x) / (7 * np.sin(x * math.pi))
    h_9 = lambda x: np.sin(9 * math.pi * x) / (9 * np.sin(x * math.pi))

    t = np.linspace(0.001, 1, 64)

    plt.plot(t, 20*np.log10(np.abs(h_3(t))), label=r'$H_3$')
    plt.plot(t, 20*np.log10(np.abs(h_5(t))), label=r'$H_5$')
    plt.plot(t, 20*np.log10(np.abs(h_7(t))), label=r'$H_7$')
    plt.plot(t, 20*np.log10(np.abs(h_9(t))), label=r'$H_9$')
    plt.xlabel('t')
    plt.ylabel(r'$H(t)$')
    plt.legend()
    plt.show()

def task2_inner():
    h_7 = lambda x: 1 / 21 * (7 + 12 * np.cos(x) + 6 * np.cos(2 * x) - 4 * np.cos(3 * x))
    h_9 = lambda x: 1 / 231 * (59 + 108 * np.cos(x) + 78 * np.cos(2 * x) + 28 * np.cos(3 * x) - 42 * np.cos(4 * x))
    h_11 = lambda x: 1 / 429 * (
            89 + 168 * np.cos(x) + 138 * np.cos(2 * x) + 88 * np.cos(3 * x) + 18 * np.cos(4 * x) - 72 * np.cos(
        5 * x))
    h_13 = lambda x: 1 / 143 * (
            25 + 48 * np.cos(x) + 42 * np.cos(2 * x) + 32 * np.cos(3 * x) + 18 * np.cos(4 * x) - 22 * np.cos(6 * x))

    t = np.linspace(0.001, 2*math.pi, 64)

    plt.plot(t, 20*np.log10(np.abs(h_7(t))), label=r'$H_7$')
    plt.plot(t, 20*np.log10(np.abs(h_9(t))), label=r'$H_9$')
    plt.plot(t, 20*np.log10(np.abs(h_11(t))), label=r'$H_{11}$')
    plt.plot(t, 20*np.log10(np.abs(h_13(t))), label=r'$H_{13}$')
    plt.xlabel('t')
    plt.ylabel(r'$H(t)$')
    plt.legend()
    plt.show()

def task3_inner():
    h_9 = lambda x: 1 / 429 * (
                179 + 270 * np.cos(x) + 60 * np.cos(2 * x) - 110 * np.cos(3 * x) + 30 * np.cos(4 * x))
    h_11 = lambda x: 1 / 429 * (
                143 + 240 * np.cos(x) + 120 * np.cos(2 * x) - 20 * np.cos(3 * x) - 90 * np.cos(4 * x) + 36 * np.cos(
            5 * x))
    h_13 = lambda x: 1 / 2431 * (677 + 1200 * np.cos(x) + 780 * np.cos(2 * x) + 220 * np.cos(3 * x) - 270 * np.cos(
        4 * x) - 396 * np.cos(5 * x) + 220 * np.cos(6 * x))
    h_15 = lambda x: 1 / 46189 * (
                11063 + 20250 * np.cos(x) + 15000 * np.cos(2 * x) + 7510 * np.cos(3 * x) - 330 * np.cos(
            4 * x) - 5874 * np.cos(5 * x) - 5720 * np.cos(6 * x) + 4290 * np.cos(7 * x))

    t = np.linspace(0.001, 2*math.pi, 64)

    plt.plot(t, 20*np.log10(np.abs(h_9(t))), label=r'$H_9$')
    plt.plot(t, 20*np.log10(np.abs(h_11(t))), label=r'$H_11$')
    plt.plot(t, 20*np.log10(np.abs(h_13(t))), label=r'$H_{13}$')
    plt.plot(t, 20*np.log10(np.abs(h_15(t))), label=r'$H_{15}$')
    plt.xlabel('t')
    plt.ylabel(r'$H(t)$')
    plt.legend()
    plt.show()

def task4_inner():
    h_21 = lambda x: 1 / 320 * (74 + 134 * np.cos(2 * math.pi * x) + 92 * np.cos(4 * math.pi * x) + 42 * np.cos(
        6 * math.pi * x) + 6 * np.cos(8 * math.pi * x) - 10 * np.cos(10 * math.pi * x) - 12 * np.cos(
        12 * math.pi * x) - 6 * np.cos(14 * math.pi * x))
    h_15 = lambda x: 1 / 350 * (60 + 114 * np.cos(2 * math.pi * x) + 94 * np.cos(4 * math.pi * x) + 66 * np.cos(
        6 * math.pi * x) + 36 * np.cos(8 * math.pi * x) + 12 * np.cos(10 * math.pi * x) - 4 * np.cos(
        12 * math.pi * x) - 10 * np.cos(14 * math.pi * x) - 10 * np.cos(16 * math.pi * x) - 6 * np.cos(
        18 * math.pi * x) - 2 * np.cos(20 * math.pi * x))

    t = np.linspace(0.001, 1, 64)

    plt.plot(t, 20*np.log10(np.abs(h_21(t))), label=r'$H_21$')
    plt.plot(t, 20*np.log10(np.abs(h_15(t))), label=r'$H_15$')
    plt.xlabel('t')
    plt.ylabel(r'$H(t)$')
    plt.legend()
    plt.show()

task1_inner()
task2_inner()
task3_inner()
task4_inner()
"""


# task7
"""
h_rect = lambda x: (1 / (2j*np.sin(math.pi*x))).imag
h_trap = lambda x: (np.cos(math.pi*x) / (2j*np.sin(math.pi*x))).imag
h_simp = lambda x: ((np.cos(2*math.pi*x)+2)/ (3j*np.sin(2*math.pi*x))).imag

y_rect = lambda x: math.pi*x / (np.sin(math.pi * x))
y_trap = lambda x: np.cos(math.pi * x)*(math.pi*x/np.sin(x*math.pi))
y_simp = lambda x: (np.cos(2*math.pi*x) + 2)/3

t = np.linspace(1e-10, 0.5, 300)

plt.plot(t, h_rect(t), label=r'$H_{rect}$')
plt.xlabel('t')
plt.ylabel(r'$H(t)$')
plt.legend()
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, -50, 10.))
plt.show()

plt.plot(t, h_trap(t), label=r'$H_{trap}$')
plt.xlabel('t')
plt.ylabel(r'$H(t)$')
plt.legend()
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, -50, 10.))
plt.show()

plt.plot(t, h_simp(t), label=r'$H_{simp}$')
plt.xlabel('t')
plt.ylabel(r'$H(t)$')
plt.legend()
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, -50, 10.))
plt.show()

plt.plot(t, y_rect(t), label=r'$y_{rect}$')
plt.plot(t, y_trap(t), label=r'$y_{trap}$')
plt.plot(t, y_simp(t), label=r'$y_{simp}$')
plt.xlabel('t')
plt.ylabel(r'$y(t)$')
plt.legend()
plt.show()
"""


# task8
"""
h = lambda x: ((np.cos(3*math.pi*x)+3*np.cos(math.pi*x))/(8j*np.sin(3*math.pi*x))).imag
y = lambda x: (1/12)*(np.cos(3*math.pi*x)+3*np.cos(math.pi*x))*((3*math.pi*x)/np.sin(3*math.pi*x))

t = np.linspace(1e-10, 0.5, 300)

plt.plot(t, h(t), label=r'$H$', )
plt.xlabel('t')
plt.ylabel(r'$H(t)$')
plt.legend()
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, -50, 10.))
plt.show()

plt.plot(t, y(t), label=r'$y$')
plt.xlabel('t')
plt.ylabel(r'$y(t)$')
plt.legend()
plt.show()
"""
