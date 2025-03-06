import numpy as np

##############################################################
# Question 1

# Initializing empty array, then adding new elements one by one.
# These new elements are random integers ranging from -100 to 100.

array_1 = np.array([])
for i in range(25):
    array_1 = np.append(array_1, np.random.randint(-100, 100))

# Easier way of doing this using numpy functions:
# array_1 = np.random.randint(-100, 100, 25)

print("Question 1:")
print('\nArray of 25 random integers between -100 and 100:\n', array_1)

##############################################################
# Question 2

sum_array = 0
for i in range(len(array_1)):
    sum_array += array_1[i]
avg_array = sum_array / len(array_1)

# Easier way of doing this using numpy functions:
# sum_array = sum(array_1)
# avg_array = np.mean(array_1)

print("\n#############################################################################")
print("Question 2:")
print('\nSum of the array:', sum_array)
print('\nAverage of the array:', avg_array)

##############################################################
# Question 3

# Initializing a 3x3 matrix of zeros, then changing the elements in a nested for loop.

matrix_1 = np.zeros((3, 3))

for i in range(np.size(matrix_1, 0)):
    for j in range(np.size(matrix_1, 1)):
        matrix_1[i, j] = np.random.rand()

# np.random.rand() specifically generates a random real number between 0 and 1.
# If you want to generate a real number between a different interval, use np.random.uniform(low, high) instead.

# Easier ways of doing this using numpy functions:
# matrix_1 = np.random.rand(3, 3), OR
# matrix_1 = np.random.randn(3, 3), OR
# matrix_1 = np.random.random((3, 3))
# See the NumPy documentation if you want to know more about their differences.

print("\n#############################################################################")
print("Question 3:")
print('\nThe 3x3 matrix:\n', matrix_1)
print('\na. Second row of the matrix:\n', matrix_1[1, :])    # matrix_1[1] would also work.
print('\nb. Third column of the matrix:\n', matrix_1[:, 2])  # here, you CANNOT omit the ':', unlike we did previously.
print('\nc. Second and third rows combined:\n', matrix_1[1:3, :])     # again, matrix_1[1:3] would also work.
print('\nd. 1st and 2nd rows, 2nd and 3rd columns:\n', matrix_1[0:2, 1:3])
print('\ne. The transpose of (c):\n', matrix_1[1:3].T)       # or, np.transpose(matrix_1[1:3])

# When we implement indexing in the 'a:b' format, we mean:
# a, a+1, a+2, ...., b-2, b-1

############################################################
# Question 4

print("\n#############################################################################")
print("Question 4:")

#   4.a
a = np.arange(1, 6, 0.5)
# Alternative: a = np.linspace(1, 6, 10, endpoint=False)
print('\na.', a)

#   4.b
b = np.arange(6, 1, -0.5)
# Alternative: b = np.linspace(6, 1, 10, endpoint=False)
print('\nb.', b)

#   4.c
v_stacked = np.vstack((a, b))
# v_stacked = np.array([a, b]) would also work identically.
print('\nc. 2x10 matrix:\n', v_stacked)

#   4.d
h_stacked = np.column_stack((a, b))
# h_stacked = np.vstack((a, b)).T would also work identically.
print('\nd. 10x2 matrix:\n', v_stacked)

#   4.e
mean_squared_error = 0
for i in range(len(a)):
    mean_squared_error += (a[i] - b[i]) ** 2
mean_squared_error /= len(a)

# A very easy way of doing this is by using the numpy "mean" function, such as:
# mean_squared_error = np.mean((a-b)**2)
print("\ne. MSE:", mean_squared_error)

# I named this last variable "mean_squared_error" because we're implementing the MSE equation here.
