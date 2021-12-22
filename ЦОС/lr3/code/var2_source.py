import numpy as np
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt

fig = plt.gcf()
fig.set_size_inches(12, 8)


def rect(orig):
    output = np.empty(len(orig))
    output[0] = 0
    for i in range(1, len(orig)):
        output[i] = output[i - 1] + orig[i]
    return output


def simpson(orig):
    output = np.empty(len(orig))
    output[0] = 0
    for i in range(1, len(orig) - 1):
        output[i] = output[i - 1] + (orig[i - 1] + orig[i] + 4 * orig[i + 1]) / 3
    return output


def trap(orig):
    output = np.empty(len(orig))
    output[0] = 0
    for i in range(1, len(orig) - 1):
        output[i] = output[i - 1] + (orig[i] + orig[i + 1]) / 2
    return output


def get_analog(n):
    ws = np.arange(0, np.pi + 0.1 * np.pi, 0.1 * np.pi)
    us = np.random.random(size=11) * 0.5
    As = np.random.randint(1, 11, 11)
    x = np.linspace(0, 32, n)
    y = np.zeros(n)
    for w, u, A in zip(ws, us, As):
        y += (A * np.cos(w * x + u))
    y = y / np.sum(As)
    return x, y


def get_discrete(y):
    out_x = np.linspace(0, 32, 32)
    out_y = []
    for idx in range(32):
        out_y.append(y[idx * 8])
    return out_x, np.array(out_y)


def gen_analog_make_discrete(numfig):
    x, y = get_analog(256)
    plt.plot(x, y, numfig)
    plt.ylabel(r'$x$(t)')
    plt.xlabel('t')
    plt.title('Аналоговый сигнал')
    plt.savefig(str(numfig) + ".png", dpi=150)
    numfig += 1
    plt.clf()

    x_disc, y_disc = get_discrete(y)
    plt.stem(x_disc, y_disc)
    plt.ylabel(r'$x$(t)')
    plt.xlabel('t')
    plt.title('Дискретный сигнал')
    plt.savefig(str(numfig) + ".png", dpi=150)
    numfig += 1
    plt.clf()

    return x_disc, y_disc


def spectrum(y, numfig):
    yf = fft(y)
    xf = fftfreq(32, 1)
    plt.stem(xf, np.abs(yf))
    plt.ylabel(r'S')
    plt.xlabel('Частота')
    plt.title('Дискретное преобразование Фурье')
    plt.savefig(str(numfig) + ".png", dpi=150)
    numfig += 1
    plt.clf()

    return


def linear_avg(x, y, numfig):
    lin_avg_5 = np.convolve(y, np.ones(5), 'same') / 5
    lin_avg_9 = np.convolve(y, np.ones(9), 'same') / 9
    plt.stem(x, y, 'r', markerfmt='ro', label='Исх. сигнал')
    plt.stem(x, lin_avg_5, 'g--', markerfmt='go', label='Лин. сглаж. по 5-ти точкам')
    plt.ylabel(r'$x$(t)')
    plt.xlabel('t')
    plt.legend()
    plt.title('Линейное сглаживание по 5-ти точкам (зел.) и исходный сигнал (красн.)')
    plt.savefig(str(numfig) + ".png", dpi=150)
    numfig += 1
    plt.clf()

    plt.stem(x, y, 'r', markerfmt='ro', label='Исх. сигнал')
    plt.stem(x, lin_avg_9, 'g--', markerfmt='go', label='Лин. сглаж. по 9-ти точкам')
    plt.ylabel(r'$x$(t)')
    plt.xlabel('t')
    plt.legend()
    plt.title('Линейное сглаживание по 9-ти точкам (зел.) и исходный сигнал (красн.)')
    plt.savefig(str(numfig) + ".png", dpi=150)
    numfig += 1
    plt.clf()

    return


def linear_avg_spectrum(x, y, numfig):
    lin_avg_5 = np.convolve(y, np.ones(5), 'same') / 5
    lin_avg_9 = np.convolve(y, np.ones(9), 'same') / 9

    xf = fftfreq(32, 1)

    f = fft(y)
    f = 2 * np.abs(f) / len(f)
    f_5 = fft(lin_avg_5)
    f_5 = 2 * np.abs(f_5) / len(f_5)
    f_9 = fft(lin_avg_9)
    f_9 = 2 * np.abs(f_9) / len(f_9)

    plt.subplot(221)
    plt.stem(xf, f)
    plt.ylabel(r'S')
    plt.xlabel('Частота')
    plt.title('Спектр (исх. сигнал)')

    plt.subplot(222)
    plt.stem(xf, f_5)
    plt.ylabel(r'S')
    plt.xlabel('Частота')
    plt.title('Спектр (лин. сглаж. по 5-ти точкам)')

    plt.subplot(223)
    plt.stem(xf, f_9)
    plt.ylabel(r'S')
    plt.xlabel('Частота')
    plt.title('Спектр (лин. сглаж. по 9-ти точкам)')
    plt.savefig(str(numfig) + ".png", dpi=150)
    numfig += 1
    plt.clf()
    return


def square_avg(x, y, numfig):
    sq_avg_5 = np.convolve(y, np.array([-3, 12, 17, 12, -3]), 'same') / 35
    sq_avg_9 = np.convolve(y, np.array([-21, 14, 39, 54, 59, 54, 39, 14, -21]), 'same') / 231

    plt.stem(x, y, 'r', markerfmt='ro', label='Исх. сигнал')
    plt.stem(x, sq_avg_5, 'g', markerfmt='go', label='Сглаж. 2-й степени (5 точек)')
    plt.stem(x, sq_avg_9, 'b', markerfmt='bo', label='Сглаж. 2-й степени (9 точек)')
    plt.xlabel('t')
    plt.ylabel(r'$y(t)$')
    plt.legend()
    plt.savefig(str(numfig) + ".png", dpi=150)
    numfig += 1
    plt.clf()

    xf = fftfreq(32, 1)
    f = fft(y)
    f = 2 * np.abs(f) / len(f)
    f_5 = fft(sq_avg_5)
    f_5 = 2 * np.abs(f_5) / len(f_5)
    f_9 = fft(sq_avg_9)
    f_9 = 2 * np.abs(f_9) / len(f_9)

    plt.subplot(221)
    plt.stem(xf, f)
    plt.ylabel(r'S')
    plt.xlabel('Частота')
    plt.title('Спектр (исх. сигнал)')

    plt.subplot(222)
    plt.stem(xf, f_5)
    plt.ylabel(r'S')
    plt.xlabel('Частота')
    plt.title('Спектр (сглаж. 2-й ст. по 5-ти точкам)')

    plt.subplot(223)
    plt.stem(xf, f_9)
    plt.ylabel(r'S')
    plt.xlabel('Частота')
    plt.title('Спектр (сглаж. 2-й ст. по 9-ти точкам)')
    plt.savefig(str(numfig) + ".png", dpi=150)
    numfig += 1
    plt.clf()


def quad_avg(x, y, numfig):
    quad_avg_7 = np.convolve(y, np.array([5, -30, 75, 131, 75, -30, 5]), 'same') / 231
    quad_avg_11 = np.convolve(y, np.array([13, -45, -10, 60, 120, 143, 120, 60, -10, -45, 13]), 'same') / 429

    plt.stem(x, y, 'r', markerfmt='ro', label='Исх. сигнал')
    plt.stem(x, quad_avg_7, 'g', markerfmt='go', label='Сглаж. 4-й степени (7 точек)')
    plt.stem(x, quad_avg_11, 'b', markerfmt='bo', label='Сглаж. 4-й степени (11 точек)')
    plt.xlabel('t')
    plt.ylabel(r'$y(t)$')
    plt.legend()
    plt.savefig(str(numfig) + ".png", dpi=150)
    numfig += 1
    plt.clf()

    xf = fftfreq(32, 1)
    f = fft(y)
    f = 2 * np.abs(f) / len(f)
    f_7 = fft(quad_avg_7)
    f_7 = 2 * np.abs(f_7) / len(f_7)
    f_11 = fft(quad_avg_11)
    f_11 = 2 * np.abs(f_11) / len(f_11)

    plt.subplot(221)
    plt.stem(xf, f)
    plt.ylabel(r'S')
    plt.xlabel('Частота')
    plt.title('Спектр (исх. сигнал)')

    plt.subplot(222)
    plt.stem(xf, f_7)
    plt.ylabel(r'S')
    plt.xlabel('Частота')
    plt.title('Спектр (сглаж. 4-й ст. по 7-ти точкам)')

    plt.subplot(223)
    plt.stem(xf, f_11)
    plt.ylabel(r'S')
    plt.xlabel('Частота')
    plt.title('Спектр (сглаж. 4-й ст. по 11-ти точкам)')
    plt.savefig(str(numfig) + ".png", dpi=150)
    numfig += 1
    plt.clf()
    return


def diff_avg(x, y, numfig):
    diff_avg = np.convolve(y, np.array([-1, 0, 1]), 'same') / 2

    plt.stem(x, y, 'r', markerfmt='ro', label='Исх. сигнал')
    plt.stem(x, diff_avg, 'g', markerfmt='go', label='Сглаж. дифф. 1-го порядка')
    plt.xlabel('t')
    plt.ylabel(r'$y(t)$')
    plt.legend()
    plt.savefig(str(numfig) + ".png", dpi=150)
    numfig += 1
    plt.clf()

    xf = fftfreq(32, 1)
    f = fft(y)
    f = 2 * np.abs(f) / len(f)
    f_1 = fft(diff_avg)
    f_1 = 2 * np.abs(f_1) / len(f_1)

    plt.subplot(211)
    plt.stem(xf, f)
    plt.ylabel(r'S')
    plt.xlabel('Частота')
    plt.title('Спектр (исх. сигнал)')

    plt.subplot(212)
    plt.stem(xf, f_1)
    plt.ylabel(r'S')
    plt.xlabel('Частота')
    plt.title('Спектр (сглаж. дифф. 1-го порядка)')
    plt.savefig(str(numfig) + ".png", dpi=150)
    numfig += 1
    plt.clf()
    return


def integral_avg(x, y, numfig):
    y_rect = rect(y)
    y_trap = trap(y)
    y_simps = simpson(y)

    plt.stem(x, y, 'r', markerfmt='ro', label='Исх. сигнал')
    plt.stem(x, y_rect, 'g', markerfmt='go', label='Сглаж. интегр. (прямоугольниками)')
    plt.stem(x, y_trap, 'b', markerfmt='bo', label='Сглаж. интегр. (трапециями)')
    plt.stem(x, y_simps, 'm', markerfmt='mo', label='Сглаж. интегр. (Симпсона)')
    plt.xlabel('t')
    plt.ylabel(r'$y(t)$')
    plt.legend()
    plt.savefig(str(numfig) + ".png", dpi=150)
    numfig += 1
    plt.clf()

    xf = fftfreq(32, 1)
    f = fft(y)
    f = 2 * np.abs(f) / len(f)
    f_rect = fft(y_rect)
    f_rect = 2 * np.abs(f_rect) / len(f_rect)
    f_trap = fft(y_trap)
    f_trap = 2 * np.abs(f_trap) / len(f_trap)
    f_simps = fft(y_simps)
    f_simps = 2 * np.abs(f_simps) / len(f_simps)

    plt.subplot(221)
    plt.stem(xf, f)
    plt.ylabel(r'S')
    plt.xlabel('Частота')
    plt.title('Спектр (исх. сигнал)')

    plt.subplot(222)
    plt.stem(xf, f_rect)
    plt.ylabel(r'S')
    plt.xlabel('Частота')
    plt.title('Спектр (сглаж. интегр. прямоуг.)')

    plt.subplot(223)
    plt.stem(xf, f_trap)
    plt.ylabel(r'S')
    plt.xlabel('Частота')
    plt.title('Спектр (сглаж. интегр. трапец.)')

    plt.subplot(224)
    plt.stem(xf, f_simps)
    plt.ylabel(r'S')
    plt.xlabel('Частота')
    plt.title('Спектр (сглаж. интегр. Симпсона)')
    plt.savefig(str(numfig) + ".png", dpi=500)
    numfig += 1
    plt.clf()
    return


numfig = 0
xx, yy = gen_analog_make_discrete(numfig)
numfig = 10
spectrum(yy, numfig)
numfig = 20
linear_avg(xx, yy, numfig)
numfig = 30
linear_avg_spectrum(xx, yy, numfig)
numfig = 40
square_avg(xx, yy, numfig)
numfig = 50
quad_avg(xx, yy, numfig)
numfig = 60
diff_avg(xx, yy, numfig)
numfig = 70
integral_avg(xx, yy, numfig)
