import numpy as np
import matplotlib.pyplot as plt
import csv

with open("Football_players.csv", encoding="Latin-1") as f:
    csv_list = list(csv.reader(f))

height_list = np.array([])
age_list = np.array([])
salary_list = np.array([])
mental_strength = np.array([])
skill = np.array([])

for row in csv_list[1:]:
    age_list = np.append(age_list, int(row[4]))
    salary_list = np.append(salary_list, int(row[8]))
    height_list = np.append(height_list, int(row[5]))
    mental_strength = np.append(mental_strength, int(row[6]))
    skill = np.append(skill, int(row[7]))

X = np.column_stack((np.ones(len(age_list)), age_list, height_list, mental_strength, skill))
y = salary_list


def calculate_multiple_regression(X, y):
    XT_X = np.dot(X.T, X)
    XT_y = np.dot(X.T, y)
    weights = np.linalg.solve(XT_X, XT_y)
    return weights


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def cross_validation_mse(X, y, k):
    n = len(X)
    indices = np.arange(n)
    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1

    current = 0
    mse_list = []

    for fold_size in fold_sizes:
        start, end = current, current + fold_size
        X_test = X[start:end]
        y_test = y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        weights = calculate_multiple_regression(X_train, y_train)
        y_pred = np.dot(X_test, weights)
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)

        current = end

    return np.mean(mse_list)



print(f"Validation MSE \t 8-fold CV MSE")
print(f"---------------\t--------------")

for i in range(10):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    split = int(len(X) * 0.8)

    X_train, X_test = X_shuffled[:split], X_shuffled[split:]
    y_train, y_test = y_shuffled[:split], y_shuffled[split:]

    weights = calculate_multiple_regression(X_train, y_train)

    y_pred = np.dot(X_test, weights)

    val_mse = mean_squared_error(y_test, y_pred)
    cv_mse = cross_validation_mse(X_shuffled, y_shuffled, k=8)

    print(f"{val_mse:.2f} \t {cv_mse:.2f}")
