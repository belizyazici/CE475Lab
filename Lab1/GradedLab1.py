import numpy as np
import matplotlib.pyplot as plt
import csv

with open("Football_players.csv", encoding="Latin-1") as f:
    csv_list = list(csv.reader(f))

height_list = np.array([])
age_list = np.array([])
salary_list = np.array([])

for row in csv_list[1:]:
    age_list = np.append(age_list, int(row[4]))
    salary_list = np.append(salary_list, int(row[8]))
    height_list = np.append(height_list, int(row[5]))


def calculate_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    b1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b0 = y_mean - b1 * x_mean

    # print(b0, b1)
    return b0, b1


def plot_regression(x, y, b0, b1, title, a):
    a.scatter(x, y, color='blue')
    y_mean = b1 * x + b0
    a.plot(x, y_mean, color='red')

    if np.array_equal(x, age_list):
        a.set_xlabel('Age')
    else:
        a.set_xlabel('Height')

    a.set_ylabel('Salary')
    a.set_title(title)


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

b0_age, b1_age = calculate_regression(age_list, salary_list)
b0_height, b1_height = calculate_regression(height_list, salary_list)

print("Model1 - B0: ", b0_age, " B1: ", b1_age)
print("Model1 - B0: ", b0_height, " B1: ", b1_height)

plot_regression(age_list, salary_list, b0_age, b1_age, 'Simple Linear Regression: Age - Salary Regression', ax[0])
plot_regression(height_list, salary_list, b0_height, b1_height, 'Linear Regression: Height - Salary Regression', ax[1])

plt.tight_layout()
plt.show()

