import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

w = np.arange(0, 1.1, 0.1) * np.pi
A = rng.integers(low=1, high=12, size=11)
phi = rng.uniform(size=11) * 0.5
s = lambda _t: sum(A * np.cos(w * _t + phi))
N = 32
x = np.linspace(0, N - 1, N * 100)
t = np.arange(N)
y = [s(_tk) for _tk in t]

# 1 - Формирование дискретного сигнала -------
plt.plot(x, [s(i) for i in x])
plt.title("Аналоговый сигнал")
plt.xlabel("t")
plt.ylabel("s(t)")
plt.grid()
plt.show()

plt.stem(t, y)
plt.title("Дискретный сигнал")
plt.xlabel("t")
plt.ylabel("s(t)")
plt.grid()
plt.show()

# --- Амплитудный спектр
N = 20
t = np.arange(20)
y = y[:20]
spectrum = lambda k: sum(y * np.exp(-2j * np.pi * t * k / N))
spectrum_x = [spectrum(k) for k in t]
plt.stem(t, np.abs(spectrum_x), basefmt="black")
plt.title("Амплитудный спектр сигнала")
omega = np.round(np.linspace(0, 2 * np.pi, N // 2), 2)
plt.xticks(range(0, N, 2), omega)
plt.xlabel("$\omega$")
plt.ylabel("$|S(\omega)|$")
plt.show()


"""N = 20
n = np.arange(N)
x = x[:20]
spectrum = lambda k: sum(x * np.exp(-2j*np.pi*n*k/N))
spectrum_x = [spectrum(k) for k in n]
plt.stem(n, np.abs(spectrum_x), basefmt="black")
plt.title("Амплитудный спектр сигнала")
omega = np.round(np.linspace(0, 2*np.pi, N//2), 2)
plt.xticks(range(0, N, 2), omega)
plt.xlabel("$\omega$")
plt.ylabel("$|S(\omega)|$")
plt.show()"""

exit()

# Функция для построения графиков
def draw_plots(fltr, filter_name, y=None):
    if y is None:
        y = [fltr(k + 20) for k in n]
    plt.stem(n, x, basefmt="black", label="$x(n)$")
    plt.stem(n, y, basefmt="black", linefmt="C3--", markerfmt="C3o-", label="$y(n)$")
    plt.xlabel("n")
    plt.xticks(range(0, N, 2))
    plt.ylabel("Сигнал")
    plt.legend()
    plt.title(filter_name)
    plt.show()

    dft_y = lambda k: sum(y * np.exp(-2j * np.pi * n * k / N))
    spectrum_y = [dft_y(k) for k in n]

    plt.stem(n, np.abs(spectrum_x), basefmt="black")
    plt.stem(n, np.abs(spectrum_y), basefmt="black", linefmt="C3--", markerfmt="C3o-")
    plt.title("Амплитудный спектр сигнала")
    plt.xticks(range(0, N, 2), omega)
    plt.xlabel("$\omega$")
    plt.ylabel("$|S(\omega)|$")
    plt.show()


# Линейное сглаживание
x_padded = list(np.tile(x, 3))
y1_5 = lambda k: sum(x_padded[(k - 2):(k + 3)]) / 5
h1_5 = lambda w: (2 * np.cos(2 * w) + 2 * np.cos(w) + 1) / 5
draw_plots(y1_5, "Сигнал, линейно сглаженный по 5 точкам")
y1_9 = lambda k: sum(x_padded[(k - 4):(k + 5)]) / 9
h1_9 = lambda w: (2 * np.cos(4 * w) + 2 * np.cos(3 * w) + 2 * np.cos(2 * w) + 2 * np.cos(w) + 1) / 9
draw_plots(y1_9, "Сигнал, линейно сглаженный по 9 точкам")
f = np.linspace(0, 2 * np.pi, 1000)
plt.plot(f, np.abs(h1_5(f)), label="5 точек")
plt.plot(f, np.abs(h1_9(f)), label="9 точек")
plt.xticks(np.round(np.linspace(0, 2 * np.pi, 7), 2))
plt.title("Передаточные функции фильтров")
plt.xlabel("$\omega$")
plt.ylabel("$H(\omega)$")
plt.legend()
plt.show()

# Сглаживание полиномом 2 степени
y2_5 = lambda k: (-3 * x_padded[k - 2] + 12 * x_padded[k - 1] + 17 * x_padded[k] +
                  12 * x_padded[k + 1] - 3 * x_padded[k + 2]) / 35
h2_5 = lambda w: (17 - 6 * np.cos(2 * w) + 24 * np.cos(w)) / 35
draw_plots(y2_5, "Сигнал, сглаженный полиномом 2-ой степени по 5 точкам")
y2_9 = lambda k: (-21 * x_padded[k - 4] + 14 * x_padded[k - 3] + 39 * x_padded[k - 2] +
                  54 * x_padded[k - 1] + 59 * x_padded[k] + 54 * x_padded[k + 1] +
                  39 * x_padded[k + 2] + 14 * x_padded[k + 3] - 21 * x_padded[k + 4]) / 231
h2_9 = lambda w: (59 - 42 * np.cos(4 * w) + 28 * np.cos(3 * w) + 78 * np.cos(2 * w)
                  + 108 * np.cos(w)) / 231
draw_plots(y2_9, "Сигнал, сглаженный полиномом 2-ой степени по 9 точкам")
plt.plot(f, np.abs(h2_5(f)), label="5 точек")
plt.plot(f, np.abs(h2_9(f)), label="9 точек")
plt.xticks(np.round(np.linspace(0, 2 * np.pi, 7), 2))
plt.title("Передаточные функции фильтров")
plt.xlabel("$\omega$")
plt.ylabel("$H(\omega)$")
plt.legend()
plt.show()

# Сглаживание полиномом 4 степени
y4_7 = lambda k: (5 * x_padded[k - 3] - 30 * x_padded[k - 2] + 75 * x_padded[k - 1] +
                  131 * x_padded[k] + 75 * x_padded[k + 1] - 30 * x_padded[k + 2] +
                  5 * x_padded[k + 3]) / 231
h4_7 = lambda w: (131 + 10 * np.cos(3 * w) - 60 * np.cos(2 * w) + 150 * np.cos(w)) / 231
draw_plots(y4_7, "Сигнал, сглаженный полиномом 4-ой степени по 7 точкам")
y4_11 = lambda k: (18 * x_padded[k - 5] - 45 * x_padded[k - 4] - 10 * x_padded[k - 3] +
                   60 * x_padded[k - 2] + 120 * x_padded[k - 1] + 143 * x_padded[k] +
                   120 * x_padded[k + 1] + 60 * x_padded[k + 2] - 10 * x_padded[k + 3] -
                   45 * x_padded[k + 4] + 18 * x_padded[k + 5]) / 429
h4_11 = lambda w: (146 + 36 * np.cos(5 * w) - 90 * np.cos(4 * w) - 20 * np.cos(3 * w) +
                   120 * np.cos(2 * w) + 240 * np.cos(w)) / 429
draw_plots(y4_11, "Сигнал, сглаженный полиномом 4-ой степени по 11 точкам")
plt.plot(f, np.abs(h4_7(f)), label="7 точек")
plt.plot(f, np.abs(h4_11(f)), label="11 точек")
plt.xticks(np.round(np.linspace(0, 2 * np.pi, 7), 2))
plt.title("Передаточные функции фильтров")
plt.xlabel("$\omega$")
plt.ylabel("$H(\omega)$")
plt.legend()
plt.show()

# Дифференцирование
y_diff = lambda k: (x_padded[k + 1] - x_padded[k - 1]) / 2
h_diff = lambda w: 1j * np.sin(w)
draw_plots(y_diff, "Сигнал, полученный в результате численного дифференцирования")
plt.plot(f, np.abs(h_diff(f)))
plt.xticks(np.round(np.linspace(0, 2 * np.pi, 7), 2))
plt.title("Передаточная функция фильтра")
plt.xlabel("$\omega$")
plt.ylabel("$H(\omega)$")
plt.show()


# Интегрирование
def y_rect():
    y = np.zeros(N)
    for i in range(1, N):
        y[i] = y[i - 1] + x[i]
    return y

draw_plots(None, "Сигнал, полученный интегрированием по формуле прямоугольников", y_rect())


def y_trap():
    y = np.zeros(N)
    for i in range(1, N - 1):
        y[i] = y[i - 1] + (x[i] + x[i + 1]) / 2
    return y


draw_plots(None, "Сигнал, полученный интегрированием по формуле трапеций", y_trap())


def y_simp():
    y = np.zeros(N)
    for i in range(1, N - 1):
        y[i + 1] = y[i - 1] + (x[i + 1] + 4 * x[i] + x[i - 1])
    return y


draw_plots(None, "Сигнал, полученный интегрированием по формуле Симпсона", y_simp())
h_rect = lambda w: (1 / (2j * np.sin(w / 2))).imag
h_trap = lambda w: (np.cos(w / 2) / (2j * np.sin(w / 2))).imag
h_simp = lambda w: ((np.cos(w) + 2) / (3j * np.sin(w))).imag
plt.plot(f, h_rect(f), label="Формула прямоугольников")
plt.plot(f, h_trap(f), label="Формула трапеций")
plt.plot(f, h_simp(f), label="Формула Симпсона")
plt.ylim(-10, 10)
plt.legend()
plt.xticks(np.round(np.linspace(0, 2 * np.pi, 7), 2))
plt.xlabel("$\omega$")
plt.ylabel("$H(\omega)$")
plt.title("Передаточная функция для формул численного интегрирования")
plt.show()
