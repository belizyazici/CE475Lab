import numpy as np
import csv
from sklearn.metrics import mean_squared_error

with open("Football_players_new.csv", encoding="Latin-1") as f:
    csv_list = list(csv.reader(f))

age_list = np.array([int(row[4]) for row in csv_list[1:]])
height_list = np.array([int(row[5]) for row in csv_list[1:]])
salary_list = np.array([int(row[8]) for row in csv_list[1:]])
mental_strength = np.array([int(row[6]) for row in csv_list[1:]])
skill = np.array([int(row[7]) for row in csv_list[1:]])

features = ['Age', 'Height', 'Mental', 'Skill']
feature_arrays = [age_list, height_list, mental_strength, skill]
y = salary_list

def calculate_multiple_regression(X, y):
    XT_X = np.dot(X.T, X)
    XT_y = np.dot(X.T, y)
    weights = np.linalg.solve(XT_X, XT_y)
    return weights

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

models = []
remaining_features = features.copy()
remaining_arrays = feature_arrays.copy()
importance_sequence = []

X_full = np.column_stack(feature_arrays)
mse_full = cross_validation_mse(X_full, y, 10)
models.append((features.copy(), mse_full))

for i in range(len(features) - 1):
    r2_scores = []
    for j in range(len(remaining_features)):
        temp_arrays = [arr for k, arr in enumerate(remaining_arrays) if k != j]
        temp_X = np.column_stack(temp_arrays)
        weights = calculate_multiple_regression(temp_X, y)
        y_pred = np.dot(temp_X, weights)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        r2_scores.append((j, r2))

    r2_scores.sort(key=lambda x: x[1])
    del_idx = r2_scores[0][0]
    importance_sequence.append(remaining_features[del_idx])
    del remaining_features[del_idx]
    del remaining_arrays[del_idx]

    X = np.column_stack(remaining_arrays)
    mse = cross_validation_mse(X, y, 10)
    models.append((remaining_features.copy(), mse))

importance_sequence.append(remaining_features[0])

print("\nModel\tFeatures Included\t\tCross-Validation MSE")
print("-------------------------------------------------------------")
for i in range(len(models)):
    f_list, mse = models[len(models) - 1 - i]
    print(f"M{i}\t{f_list}\t{mse:.2f}")

best_model = min(models, key=lambda x: x[1])
best_index = len(models) - 1 - models.index(best_model)
print(f"\nMost optimal model: M{best_index}")
print(f"Features in optimal model: {best_model[0]}")
print(f"Feature importance order (high to low): {importance_sequence[::-1]}")


