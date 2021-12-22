import numpy as np
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt


def init_signals():
    rng = np.random.default_rng()

    w = np.arange(0, 1.1, 0.1) * np.pi
    A = rng.integers(low=1, high=12, size=11)
    phi = rng.uniform(size=11) * 0.5
    s = lambda _t: sum(A * np.cos(w * _t + phi))
    N = 32
    t = np.linspace(0, N - 1, N * 100)
    x = np.arange(N)
    y = [s(_tk) for _tk in x]

    plt.plot(t, [s(i) for i in t])
    plt.title("Аналоговый сигнал")
    plt.xlabel("t")
    plt.ylabel("s(t)")
    plt.grid()
    plt.show()

    plt.stem(x, y)
    plt.title("Дискретный сигнал")
    plt.xlabel("t")
    plt.ylabel("s(t)")
    plt.grid()
    plt.show()

    return x, y


def spectr(y):
    yf = fft(y)
    xf = fftfreq(32, 1)
    plt.stem(xf, np.abs(yf))
    plt.xlabel("$\omega$")
    plt.ylabel("$|S(\omega)|$")
    plt.title('Дискретное преобразование Фурье')
    plt.show()
    plt.clf()


def pol1(x, y):
    lin_avg_5 = np.convolve(y, np.ones(5), 'same') / 5
    lin_avg_9 = np.convolve(y, np.ones(9), 'same') / 9
    plt.stem(x, y, 'r', markerfmt='ro', label='Исходный')
    plt.stem(x, lin_avg_5, 'b--', markerfmt='bo', label='Сглаженный')
    plt.ylabel(r'$x$(t)')
    plt.xlabel('t')
    plt.legend()
    plt.title('Линейное сглаживание по 5 точкам')
    plt.show()
    plt.clf()

    plt.stem(x, y, 'r', markerfmt='ro', label='Исходный')
    plt.stem(x, lin_avg_9, 'b--', markerfmt='bo', label='Сглаженный')
    plt.ylabel(r'$x$(t)')
    plt.xlabel('t')
    plt.legend()
    plt.title('Линейное сглаживание по 9 точкам')
    plt.show()
    plt.clf()

    xf = fftfreq(32, 1)
    f = fft(y)
    f = 2 * np.abs(f) / len(f)
    f_5 = fft(lin_avg_5)
    f_5 = 2 * np.abs(f_5) / len(f_5)
    f_9 = fft(lin_avg_9)
    f_9 = 2 * np.abs(f_9) / len(f_9)

    def _add_default_spectr():
        plt.stem(xf, f, 'r', markerfmt='ro', label="Исходный")
        plt.xlabel("$\omega$")
        plt.ylabel("$|S(\omega)|$")

    _add_default_spectr()
    plt.stem(xf, f_5, 'b--', markerfmt='bo', label="Сглаженный")
    plt.title('Спектр лин. сглаживания по 5 точкам')
    plt.legend()
    plt.show()

    _add_default_spectr()
    plt.stem(xf, f_9, 'b--', markerfmt='bo', label="Сглаженный")
    plt.title('Спектр лин. сглаживания по 9 точкам')
    plt.legend()
    plt.show()


def pol2(x, y):
    sq_avg_5 = np.convolve(y, np.array([-3, 12, 17, 12, -3]), 'same') / 35
    sq_avg_9 = np.convolve(y, np.array([-21, 14, 39, 54, 59, 54, 39, 14, -21]), 'same') / 231

    def _add_default():
        plt.stem(x, y, 'r', markerfmt='ro', label='Исходный')
        plt.xlabel('t')
        plt.ylabel(r'$y(t)$')

    _add_default()
    plt.stem(x, sq_avg_5, 'b--', markerfmt='bo', label='Сглаженный')
    plt.legend()
    plt.title("Сглаживание пол.2 по 5 точкам")
    plt.show()

    _add_default()
    plt.stem(x, sq_avg_9, 'b--', markerfmt='bo', label='Сглаженный')
    plt.legend()
    plt.title("Сглаживание пол.2 по 9 точкам")
    plt.show()

    xf = fftfreq(32, 1)
    f = fft(y)
    f = 2 * np.abs(f) / len(f)
    f_5 = fft(sq_avg_5)
    f_5 = 2 * np.abs(f_5) / len(f_5)
    f_9 = fft(sq_avg_9)
    f_9 = 2 * np.abs(f_9) / len(f_9)

    def _add_default_spectr():
        plt.stem(xf, f, 'r', markerfmt='ro', label="Исходный")
        plt.xlabel("$\omega$")
        plt.ylabel("$|S(\omega)|$")

    _add_default_spectr()
    plt.stem(xf, f_5, 'b--', markerfmt='bo', label="Сглаженный")
    plt.title('Спектр сглаживания пол.2 по 5 точкам')
    plt.legend()
    plt.show()

    _add_default_spectr()
    plt.stem(xf, f_9, 'b--', markerfmt='bo', label="Сглаженный")
    plt.title('Спектр сглаживания пол.2 по 9 точкам')
    plt.legend()
    plt.show()


def pol4(x, y):
    quad_avg_7 = np.convolve(y, np.array([5, -30, 75, 131, 75, -30, 5]), 'same') / 231
    quad_avg_11 = np.convolve(y, np.array([13, -45, -10, 60, 120, 143, 120, 60, -10, -45, 13]), 'same') / 429

    def _add_default():
        plt.stem(x, y, 'r', markerfmt='ro', label='Исходный')
        plt.xlabel('t')
        plt.ylabel(r'$y(t)$')

    _add_default()
    plt.stem(x, quad_avg_7, 'b--', markerfmt='bo', label='Сглаженный')
    plt.title("Сглаживание пол.4 по 7 точкам")
    plt.legend()
    plt.show()

    _add_default()
    plt.stem(x, quad_avg_11, 'b--', markerfmt='bo', label='Сглаженный')
    plt.title("Сглаживание пол.4 по 11 точкам")
    plt.legend()
    plt.show()

    xf = fftfreq(32, 1)
    f = fft(y)
    f = 2 * np.abs(f) / len(f)
    f_7 = fft(quad_avg_7)
    f_7 = 2 * np.abs(f_7) / len(f_7)
    f_11 = fft(quad_avg_11)
    f_11 = 2 * np.abs(f_11) / len(f_11)

    def _add_default_spectr():
        plt.stem(xf, f, 'r', label="Исходный", markerfmt='ro')
        plt.xlabel("$\omega$")
        plt.ylabel("$|S(\omega)|$")

    _add_default_spectr()
    plt.stem(xf, f_7, 'b--', markerfmt='bo', label='Сглаженный')
    plt.legend()
    plt.title('Спектр сглаживания пол.4 по 7 точкам')
    plt.show()

    _add_default_spectr()
    plt.stem(xf, f_11, 'b--', markerfmt='bo', label='Сглаженный')
    plt.legend()
    plt.title('Спектр сглаживания пол.4 по 11 точкам')
    plt.show()


def diff1(x, y):
    diff_avg = np.convolve(y, np.array([-1, 0, 1]), 'same') / 2

    plt.stem(x, y, 'r', markerfmt='ro', label='Исходный')
    plt.stem(x, diff_avg, 'b--', markerfmt='bo', label='Сглаженный')
    plt.title("Сглаживание дифференцированием 1 порядка")
    plt.xlabel('t')
    plt.ylabel(r'$y(t)$')
    plt.legend()
    plt.show()

    xf = fftfreq(32, 1)
    f = fft(y)
    f = 2 * np.abs(f) / len(f)
    f_1 = fft(diff_avg)
    f_1 = 2 * np.abs(f_1) / len(f_1)

    plt.stem(xf, f, 'r', markerfmt='ro', label='Исходный')
    plt.stem(xf, f_1, 'b--', markerfmt='bo', label='Сглаженный')
    plt.xlabel("$\omega$")
    plt.ylabel("$|S(\omega)|$")
    plt.legend()
    plt.title('Спектр сглаживания дифф. 1 порядка')
    plt.show()


def integral_all(x, y):
    y_rect = np.empty(len(y))
    y_rect[0] = 0
    for i in range(1, len(y)):
        y_rect[i] = y_rect[i - 1] + y[i]

    y_trap = np.empty(len(y))
    y_trap[0] = 0
    for i in range(1, len(y) - 1):
        y_trap[i] = y_trap[i - 1] + (y[i] + y[i + 1]) / 2

    y_simps = np.empty(len(y))
    y_simps[0] = 0
    for i in range(1, len(y) - 1):
        y_simps[i] = y_simps[i - 1] + (y[i - 1] + y[i] + 4 * y[i + 1]) / 3

    def _add_default():
        plt.stem(x, y, 'r', markerfmt='ro', label='Исходный')
        plt.xlabel('t')
        plt.ylabel(r'$y(t)$')

    _add_default()
    plt.stem(x, y_rect, 'b--', markerfmt='bo', label='Сглаженный')
    plt.title("Сглаживание интегрированием прямоугольниками")
    plt.legend()
    plt.show()

    _add_default()
    plt.stem(x, y_trap, 'b--', markerfmt='bo', label='Сглаженный')
    plt.title("Сглаживание интегрированием трапециями")
    plt.legend()
    plt.show()

    _add_default()
    plt.stem(x, y_simps, 'b--', markerfmt='bo', label='Сглаженный')
    plt.title("Сглаживание интегрированием ф. Симпсона")
    plt.legend()
    plt.show()

    xf = fftfreq(32, 1)
    f = fft(y)
    f = 2 * np.abs(f) / len(f)
    f_rect = fft(y_rect)
    f_rect = 2 * np.abs(f_rect) / len(f_rect)
    f_trap = fft(y_trap)
    f_trap = 2 * np.abs(f_trap) / len(f_trap)
    f_simps = fft(y_simps)
    f_simps = 2 * np.abs(f_simps) / len(f_simps)

    def _add_default_spectr():
        plt.stem(xf, f, 'r', markerfmt='ro', label='Исходный')
        plt.xlabel("$\omega$")
        plt.ylabel("$|S(\omega)|$")

    _add_default_spectr()
    plt.stem(xf, f_rect, 'b--', markerfmt='bo', label='Сглаженный')
    plt.title('Спектр сглаживания инт. прямоугольниками')
    plt.legend()
    plt.show()

    _add_default_spectr()
    plt.stem(xf, f_trap, 'b--', markerfmt='bo', label='Сглаженный')
    plt.title('Спектр сглаживания инт. трапециями')
    plt.legend()
    plt.show()

    _add_default_spectr()
    plt.stem(xf, f_simps, 'b--', markerfmt='bo', label='Сглаженный')
    plt.title('Спектр сглаживания инт. ф. Симпсона')
    plt.legend()
    plt.show()


xx, yy = init_signals()
spectr(yy)
pol1(xx, yy)
pol2(xx, yy)
pol4(xx, yy)
diff1(xx, yy)
integral_all(xx, yy)