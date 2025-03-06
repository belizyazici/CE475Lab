import matplotlib.pyplot as plt
import numpy as np
import random

###################################################################################################################
# Question 1

matrix_1 = np.random.randint(0, 100, (20, 4))

print("Question 1:")
print('\nThe 20x4 matrix:\n', matrix_1)

#   1.a
matrix_1 = np.delete(matrix_1, 0, 0)
print('\na. First row deleted:\n', matrix_1)

#   1.b
matrix_1 = np.delete(matrix_1, len(matrix_1)-1, 0)
print('\nb. Last row deleted:\n', matrix_1)

#   1.c
matrix_1 = np.delete(matrix_1, range(1, 5), 0)
print('\nc. Rows 2, 3, 4 and 5 deleted:\n', matrix_1)
# You can see that the second argument can be an array containing indices.
# In that case, those indices will be deleted.
# For example, matrix_1 = np.delete(matrix_1, [1, 3, 5], 0) deletes the 2nd, 4th and 6th rows.
# Since range(1,5) = [1, 2, 3, 4], we delete the 2nd, 3rd, 4th and 5th rows.

#   1.d
matrix_1 = np.delete(matrix_1, 1, 1)
print('\nd. Second column deleted:\n', matrix_1)
# As shown, the third argument indicates whether a ROW or a COLUMN should be deleted.
# If it's 0, we delete rows; if it's 1, we delete columns.

#   1.e
matrix_1 = np.column_stack((matrix_1, np.linspace(1, 14, 14)))
print('\ne. Extra column of integers from 1 to 14 appended:\n', matrix_1)


###################################################################################################################
# Question 2

first_col = matrix_1[:, 0]
last_col = matrix_1[:, matrix_1.shape[1]-1]

# matrix_1.shape returns a tuple containing the dimensions of the multi-dimensional array.
# In this case, matrix_1.shape returns [14, 4].
# I needed the number of columns (to be able to access the last column), so we took the second value of this tuple.

print('\nFirst column:', first_col)
print('\nLast column:', last_col)

plt.scatter(last_col, first_col)

###################################################################################################################
# Question 3

y_mean = np.mean(first_col)

y_axis_values = np.ones((len(first_col)))   # Created an array of 1's, with 14 elements.
y_axis_values *= y_mean                     # Now every element of this array is equal to y_mean.

plt.plot(last_col, y_axis_values, 'r')
plt.show()

# We didn't have to create 14-element arrays, 2 elements would suffice since we're simply drawing a horizontal line.
# For example, instead of this plot line, use the following, and you will get the same result:
# plt.plot([1, 14], [y_mean, y_mean], 'r')

# But if you make x-axis values too small, the line will be very tiny.
# Similarly, if you make x-axis values too large,
#   the line will be too big and the scatter plot will be harder to interpret.

# You can try the following two lines of code to see the result:
# plt.plot([-100, 100], [y_mean, y_mean], 'r')
# plt.plot([0, 1], [y_mean, y_mean], 'r')

# matplotlib function to create horizontal lines:
# plt.hlines(y_mean, xmin=min(last_col), xmax=max(last_col))

#################################################################################################################
# Question 4

print("\n#############################################################################")
print("Question 4:")

# y_axis values of the scatter plot is stored in first_col.

squared_error_sum = 0
for i in range(len(first_col)):
    squared_error_sum += (first_col[i] - y_mean) ** 2
mean_squared_error = squared_error_sum / len(first_col)

print('MSE:', mean_squared_error)

# Remember that the y_axis values of the line consists of an array where every one of its elements is equal to y_mean.
# Since our line has the same number of elements as first_col, We could easily do the same calculation using:
# mean_squared_error = np.mean((first_col-y_axis_values)**2)


#################################################################################################################
# Question 5


x = np.linspace(-3, 3, 1000)
y = np.zeros(1000)

for i in range(len(x)):
    if x[i] == 0:
        y[i] = 0
    elif x[i] < -1 or x[i] > 1:
        y[i] = x[i]*x[i]*x[i]
    else:
        y[i] = 1/x[i]


plt.plot(x, y)
plt.xticks(np.arange(-3, 3, 0.5))   # Arranging ticks to be increments of 0.5
plt.show()
