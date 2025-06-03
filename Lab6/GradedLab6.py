import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


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

X = np.column_stack((age_list, height_list, mental_strength, skill))
y = salary_list

results = []

for n in range(10, 101, 10):
    for d in range(3, 13):
        try:
            reg = RandomForestRegressor(n_estimators=n, max_depth=d, oob_score=True, bootstrap=True, random_state=42)
            reg.fit(X, y)
            oob_pred = reg.oob_prediction_
            valid = ~np.isnan(oob_pred)
            mse = mean_squared_error(y[valid], oob_pred[valid])
            results.append((n, d, mse))
            print(f"n: {n}, d: {d}, MSE: {mse:.2f}")
        except ValueError:
            continue

ns, ds, mses = zip(*results)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ns, ds, mses, c='red', label='MSE values')

min_idx = np.argmin(mses)
ax.scatter(ns[min_idx], ds[min_idx], mses[min_idx], c='blue', label='Min MSE')

ax.set_xlabel("Number of Trees")
ax.set_ylabel("Depth")
ax.set_zlabel("MSE")
ax.set_title("Random Forest - Parameter Optimization")
plt.ticklabel_format(style="plain")
plt.show()

print("\nAverage MSE by tree depth:")
for d in range(3, 13):
    mse_d = [mse for n, depth, mse in results if depth == d]
    if mse_d:
        print(f"Depth {d}: Avg MSE = {np.mean(mse_d):.2f}")