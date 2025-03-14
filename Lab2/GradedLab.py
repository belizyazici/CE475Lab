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

train_size = len(age_list) - 20
age_train, age_test = age_list[:train_size], age_list[train_size:]
height_train, height_test = height_list[:train_size], height_list[train_size:]
salary_train, salary_test = salary_list[:train_size], salary_list[train_size:]


def calculate_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    b1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b0 = y_mean - b1 * x_mean

    # print(b0, b1)
    return b0, b1


def plot_regression(x_train, y_train, x_test, y_test, b0, b1, title, a):
    a.scatter(x_train, y_train, color='blue', label='Train Data')
    a.scatter(x_test, y_test, color='red', label='Test Data')

    y_pred = b1 * np.concatenate((x_train, x_test)) + b0
    a.plot(np.concatenate((x_train, x_test)), y_pred, color='black', label='Regression Line')

    if np.array_equal(x_train, age_train):
        a.set_xlabel('Age')
    else:
        a.set_xlabel('Height')

    a.set_ylabel('Salary')
    a.set_title(title)
    a.legend()


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

b0_age, b1_age = calculate_regression(age_train, salary_train)
b0_height, b1_height = calculate_regression(height_train, salary_train)

print("Model1 - B0: ", b0_age, " B1: ", b1_age)
print("Model1 - B0: ", b0_height, " B1: ", b1_height)

plot_regression(age_train, salary_train, age_test, salary_test, b0_age, b1_age, 'Simple Linear Regression: Age - Salary', ax[0])
plot_regression(height_train, salary_train, height_test, salary_test, b0_height, b1_height, 'Simple Linear Regression: Height - Salary', ax[1])


plt.tight_layout()
plt.show()

