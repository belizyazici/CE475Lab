import random

rand_array = []
count_odd = 0
count_even = 0
total = 0

for _ in range(25):
    value = random.randint(-100, 100)
    rand_array.append(value)

    total += value

    if value % 2 == 0:
        count_even += 1

    else:
        count_odd += 1

avg = total / 25
print("Sum: " + str(total))
print("Avg: " + str(avg))
print("Even num count: " + str(count_even))
print("Odd num count: " + str(count_odd))

print("**********************************************************************************")
import numpy as np

size = 3
matrix = np.array([[0] * size for _ in range(size)])

for i in range(size):
    for j in range(size):
        matrix[i][j] = random.randint(0, 1)

# transpose = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))] normal matrix için

for row in matrix:
    print(row)

print("Second row: ", matrix[1:2])
print("Third column: ", matrix[:, 2]) # [row[2] for row in matrix] eğer normal matrix olsaydı ama numpy için bunu yap
print("2nd and 3rd rows combined (a 2x3 matrix):\n ", matrix[1:3, :]) # son 2 satır
print("1st and 2nd rows; and 2nd and 3rd columns (a 2x2 matrix):\n ", matrix[:2, 1:]) # ilk 2 rows, 1 dışında diğer cols
print("The transpose of (c): \n", matrix[1:3, :].T)

print("**************************************************************************")


a = np.linspace(1, 6, 10) # 10 linearly spaced, ascending numbers between 1 and 6 (exclusive)
b = np.linspace(6, 1, 10) # 10 linearly spaced, descending numbers between 6 and 1


matrix2 = np.vstack((a, b))
print("Combine them to create a 2x10 matrix:\n ", matrix2)

matrix3 = np.hstack((a.reshape(-1, 1), b.reshape(-1, 1)))
print("Combine them to create a 10x2 matrix:\n ", matrix3)

result = np.sum((a - b) ** 2) / 10

print("Result: ", result)
