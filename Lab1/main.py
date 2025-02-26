import numpy as np
import matplotlib.pyplot as plt
import collections
from scipy import stats

# -------------------------
# 1. ВЫБОРКА A (дискретная)
# -------------------------

sampleA = [5, 3, 3, 3, 5, 4, 5, 3, 3, 4, 2, 1, 5,
           2, 4, 0, 2, 2, 3, 2, 1, 3, 3, 1, 2, 4,
           6, 6, 4, 1, 2, 4, 3, 1, 5, 2, 4, 6, 3,
           8, 4, 5, 1, 1, 2, 0, 2, 3, 3, 2, 4, 2,
           1, 2, 3, 1, 2, 4, 3, 0, 6, 3, 1, 4, 3, 7,
           1, 1, 0, 2, 3, 1, 1]

a = np.array(sampleA)

# 1) Min, Max, Range
a_min = a.min()
a_max = a.max()
a_range = a_max - a_min
print("Выборка A:")
print("Минимум:", a_min)
print("Максимум:", a_max)
print("Размах:", a_range)

# 2) Статистический ряд и полигон
freq_dict = collections.Counter(a)
x_vals = sorted(freq_dict.keys())
counts = [freq_dict[x] for x in x_vals]

print("\nСтатистический ряд (значение - частота):")
for val, cnt in zip(x_vals, counts):
    print(f"{val}: {cnt}")

plt.figure(figsize=(6,4))
plt.title("Полигон частот для выборки A")
plt.plot(x_vals, counts, marker='o')
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.grid(True)
plt.show()

# 3) Эмпирическая функция распределения
n = len(a)
sorted_a = np.sort(a)

empirical_cdf_x = []
empirical_cdf_y = []
current_sum = 0
last_val = None

unique_vals = np.unique(sorted_a)
for val in unique_vals:
    cnt = np.sum(sorted_a == val)
    if last_val is None:
        empirical_cdf_x.append(val)
        empirical_cdf_y.append(0)
    else:
        empirical_cdf_x.append(val)
        empirical_cdf_y.append(empirical_cdf_y[-1])
    current_sum += cnt
    F_val = current_sum / n
    empirical_cdf_x.append(val)
    empirical_cdf_y.append(F_val)
    last_val = val

plt.figure(figsize=(6,4))
plt.title("Эмпирическая функция распределения (CDF) для выборки A")
plt.step(empirical_cdf_x, empirical_cdf_y, where='post')
plt.xlabel("x")
plt.ylabel("F_n(x)")
plt.grid(True)
plt.show()

# 4) Начальные и центральные моменты
m1p = np.mean(a**1)
m2p = np.mean(a**2)
m3p = np.mean(a**3)
m4p = np.mean(a**4)

mean_a = m1p
m1 = np.mean((a - mean_a)**1)
m2 = np.mean((a - mean_a)**2)
m3 = np.mean((a - mean_a)**3)
m4 = np.mean((a - mean_a)**4)

print("\nНачальные моменты до 4-го порядка:")
print(f"m1' = {m1p:.4f}")
print(f"m2' = {m2p:.4f}")
print(f"m3' = {m3p:.4f}")
print(f"m4' = {m4p:.4f}")

print("\nЦентральные моменты до 4-го порядка:")
print(f"m1 = {m1:.4f}")  # ~0
print(f"m2 = {m2:.4f}")
print(f"m3 = {m3:.4f}")
print(f"m4 = {m4:.4f}")

# 5) Мода, медиана, асимметрия, эксцесс
# Мода (дискретная)
mode_val = max(freq_dict, key=freq_dict.get)
median_val = np.median(a)

sk = m3 / (m2**1.5)
ex = (m4 / (m2**2)) - 3

print(f"\nМода: {mode_val}")
print(f"Медиана: {median_val}")
print(f"Асимметрия (Sk): {sk:.4f}")
print(f"Эксцесс (Ex): {ex:.4f}")

# -------------------------
# 2. ВЫБОРКА B (интервальная)
# -------------------------

sampleB = [
    52, 40, 47, 54, 40, 54, 41, 74, 45, 45, 51, 76, 58, 37, 40,
    42, 53, 54, 65, 46, 65, 61, 55, 38, 66, 42, 56, 54, 40, 60,
    43, 49, 77, 64, 53, 64, 58, 54, 56, 53, 43, 35, 56, 34, 59,
    58, 66, 49, 49, 57, 48, 42, 46, 52, 59, 50, 62, 50, 55, 55,
    46, 53, 51, 50, 60, 30, 48, 56, 29, 74, 52, 60, 44, 62, 23,
    54, 40, 33, 20, 55, 42, 61, 54, 41, 45, 75, 59, 41, 51, 45,
    54, 52, 62, 69, 65, 49, 48, 63, 52, 46, 44, 55, 60, 54, 39,
    82, 67, 68, 34, 56, 51, 56, 48, 53, 47, 59, 51, 59, 66, 48,
    61, 42, 54, 33, 39, 47, 46, 47, 73, 63, 34, 44, 51, 46, 40,
    43, 30, 60, 61, 53, 47, 42, 56, 70, 48, 45, 65, 48, 48, 51,
    40, 57, 56, 33, 44, 43, 45, 35, 35, 56, 59, 66, 56, 52, 44,
    53, 49, 55, 25, 53, 48, 73, 38, 58, 72, 57, 46, 54, 55, 59,
    38, 53, 48, 68, 36, 53, 41, 55, 51, 50, 45, 50, 29, 60, 39,
    50, 59, 33, 56, 49, 31, 70, 56, 56
]

b = np.array(sampleB)

b_min = b.min()
b_max = b.max()
b_range = b_max - b_min

print("Выборка B:")
print("Минимум:", b_min)
print("Максимум:", b_max)
print("Размах:", b_range)

n_b = len(b)
k_sturges = int(1 + 3.322 * np.log10(n_b))
print("Число интервалов (Старджесс):", k_sturges)
h = b_range / k_sturges
print("Длина интервала:", h)

# Построим интервальный ряд
intervals = []
freqs = []
left = b_min
for i in range(k_sturges):
    right = left + h
    if i == k_sturges - 1:
        right = b_max
    # считаем, сколько наблюдений попало
    if i == k_sturges - 1:
        count_in_interval = np.sum((b >= left) & (b <= right))
    else:
        count_in_interval = np.sum((b >= left) & (b < right))
    intervals.append((left, right))
    freqs.append(count_in_interval)
    left = right

print("\nИнтервальный ряд:")
for (l, r), f in zip(intervals, freqs):
    print(f"[{l:.2f}, {r:.2f}): {f}")

# Гистограмма вручную
rel_freqs = [f/n_b for f in freqs]
hist_heights = [f_ / h for f_ in rel_freqs]  # частота / ширина
mid_points = [(iv[0] + iv[1]) / 2 for iv in intervals]

plt.figure(figsize=(8,4))
plt.title("Гистограмма и полигон (выборка B)")
for (l, r), hh in zip(intervals, hist_heights):
    plt.bar(l, hh, width=(r-l), align='edge', edgecolor='black', alpha=0.5)

plt.plot(mid_points, hist_heights, marker='o', color='red')
plt.xlabel("Значение")
plt.ylabel("Плотность")
plt.grid(True)
plt.show()

# Эмпирическая функция распределения
sorted_b = np.sort(b)
F_x = []
F_y = []
curr = 0
for val in sorted_b:
    curr += 1
    F_x.append(val)
    F_y.append(curr/n_b)

plt.figure(figsize=(6,4))
plt.title("Эмпирическая функция распределения (CDF) для B")
plt.step(F_x, F_y, where='post')
plt.xlabel("x")
plt.ylabel("F_n(x)")
plt.grid(True)
plt.show()

# Кумулята (можно то же самое, либо в %)
# ...

# Моменты
m1p_b = np.mean(b)
m2p_b = np.mean(b**2)
m3p_b = np.mean(b**3)
m4p_b = np.mean(b**4)

mean_b = m1p_b
m1_b = np.mean((b - mean_b)**1)
m2_b = np.mean((b - mean_b)**2)
m3_b = np.mean((b - mean_b)**3)
m4_b = np.mean((b - mean_b)**4)

sk_b = m3_b / (m2_b**1.5)
ex_b = (m4_b / (m2_b**2)) - 3

# Мода - для непрерывного часто берут интервал с макс. высотой гистограммы.
# Для простоты - через stats.mode (дискретный подход)
mode_b = stats.mode(b, keepdims=True)[0][0]
median_b = np.median(b)

print("\nМоменты B:")
print(f"m1' = {m1p_b:.4f}")
print(f"m2' = {m2p_b:.4f}")
print(f"m3' = {m3p_b:.4f}")
print(f"m4' = {m4p_b:.4f}")

print("\nЦентральные моменты B:")
print(f"m1 = {m1_b:.4f}")
print(f"m2 = {m2_b:.4f}")
print(f"m3 = {m3_b:.4f}")
print(f"m4 = {m4_b:.4f}")

print(f"\nМода (stats.mode) = {mode_b}")
print(f"Медиана = {median_b}")
print(f"Асимметрия (Sk) = {sk_b:.4f}")
print(f"Эксцесс (Ex) = {ex_b:.4f}")
