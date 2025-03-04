import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

matrix = np.array([[random.randint(0, 100) for _ in range(4)] for _ in range(20)])

print("Original one:\n ", matrix)

df_matrix = pd.DataFrame(matrix)

df_matrix.drop(df_matrix.index[0], axis=0, inplace=True)
print("Matrix after removing first row:\n", df_matrix)

df_matrix.drop(df_matrix.tail(1).index, axis=0, inplace=True)
print("Matrix after removing last row:\n", df_matrix)

df_matrix.drop(df_matrix.index[1:5], axis=0, inplace=True)
print("Matrix after removing rows 2, 3 4 and 5:\n", df_matrix)

df_matrix.drop(df_matrix.columns[1], axis=1, inplace=True) # df_matrix.columns, Pandas DataFrame’in sütun isimlerini verir (str)
print("Matrix after removing 2nd column:\n", df_matrix)

#.drop() varsayılan olarak kalıcı değişiklik yapmaz çünkü inplace=True parametresi
# eklenmezse orijinal DataFrame değişmez. Eğer değişikliklerin kalıcı olmasını
# istiyorsan inplace=True ekleyebilir veya sonucu tekrar df_matrix değişkenine
# atayabilirsin

new_column = np.linspace(1, 14, 14)                     # .shape[0] matrixin satır sayısını verir
df_matrix.insert(df_matrix.shape[1], "New", new_column) # .shape[1] matrixin sütun sayısını verir
print("Append a column of 14 linearly spaced integers from 1 to 14 (inclusive):\n", df_matrix)

print("------------------------------------------------")

first_col = df_matrix.iloc[:, 0]
last_col = df_matrix.iloc[:, -1]

plt.scatter(last_col, first_col, label="Scatterplot", color= "blue")
plt.xlabel("last col")
plt.ylabel("first col")
plt.title("Scatter Plot of First vs Last Column")
#plt.legend()
#plt.show()

print("------------------------------------------------")

total= 0

for i in first_col:
    total += i

y_mean = total / len(first_col)

print("Total: ", total)
print("Mean: ", y_mean)


plt.figure()
plt.plot(last_col, np.full_like(first_col, y_mean), color="red", label="Line plot in red")
plt.xlabel("last col")
plt.ylabel("Y Mean")
#plt.show()

first_col, y_mean

for i in first_col:
    result = ((y_mean - i) ** 2) / len(first_col)

print("MSE: ", result)

print("--------------------------------------------------")


x1 = np.linspace(-3, -1, 400)  # x ≤ -1 aralığı
x2 = np.linspace(-1, 1, 400)   # -1 < x ≤ 1 aralığı (0 hariç)
x3 = np.linspace(1, 3, 400)    # x > 1 aralığı

y1 = x1 ** 3
y2 = 1 / x2
y3 = x3 ** 3

plt.figure(figsize=(8, 6))

plt.plot(x1, y1, color="blue", label=r"$y = x^3$ for $x \leq -1$")
plt.plot(x2, y2, color="green", label=r"$y = \frac{1}{x}$ for $-1 < x \leq 1$")
plt.plot(x3, y3, color="blue")

# 0 noktasındaki tek durum
plt.scatter(0, 0, color="red", zorder=3, label="y = 0 for x = 0")

plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Piecewise Function Plot")
plt.legend()
plt.grid(True)

plt.show()
