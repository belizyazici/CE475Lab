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

train_size = int(len(age_list) * 0.8)
X_train = np.column_stack((np.ones(train_size), age_list[:train_size], height_list[:train_size], mental_strength[:train_size], skill[:train_size]))
X_test = np.column_stack((np.ones(len(age_list) - train_size), age_list[train_size:], height_list[train_size:], mental_strength[train_size:], skill[train_size:]))
Y_train = salary_list[:train_size]
Y_test = salary_list[train_size:]


def calculate_multiple_regression(X, Y):
    XT_X = np.dot(X.T, X)
    XT_Y = np.dot(X.T, Y)
    weights = np.linalg.solve(XT_X, XT_Y)
    return weights


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


X = np.column_stack((np.ones(len(age_list)), age_list, height_list, mental_strength, skill))
weights = calculate_multiple_regression(X_train, Y_train)
Y_pred = np.dot(X, weights)
mse = mean_squared_error(salary_list, Y_pred)
Y_pred_train = np.dot(X_train, weights)
Y_pred_test = np.dot(X_test, weights)

mse_train = mean_squared_error(Y_train, Y_pred_train)
mse_test = mean_squared_error(Y_test, Y_pred_test)


random_feature = np.random.randint(-1000, 1000, size=len(age_list))
X_train_new = np.column_stack((X_train, random_feature[:train_size]))
X_test_new = np.column_stack((X_test, random_feature[train_size:]))

weights_new = calculate_multiple_regression(X_train_new, Y_train)
Y_pred_test_new = np.dot(X_test_new, weights_new)
mse_test_new = mean_squared_error(Y_test, Y_pred_test_new)
mse_train_new = mean_squared_error(Y_train, np.dot(X_train_new, weights_new))

X_all = np.column_stack((np.ones(len(age_list)), age_list, height_list, mental_strength, skill))
Y_all = salary_list

weights_all = calculate_multiple_regression(X_all, Y_all)

Y_pred_all = np.dot(X_all, weights_all)

mse_all = mean_squared_error(Y_all, Y_pred_all)

print("---------------- ORIGINAL DATA ----------------")
print(f"\nTest error w/ 80-20 split\nMSE: {mse_test}")
print(f"Without split (all data as train/test) MSE: {mse_all}")
print("\n---------------- DATA W/ RANDOM COLUMN ----------------")
print(f"\nTest error w/ 80-20 split:\nMSE: {mse_test_new}")
print(f"\nTraining error:\nMSE: {mse_train_new}")

