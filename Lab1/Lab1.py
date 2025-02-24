import numpy as np
import matplotlib.pyplot as plt
import csv

with open("Football_players.csv", encoding="Latin-1") as f: # veri Türkçe karakter vs içeriyorsa encoding kullanılabilir sorunu önlemek için
    csv_list = list(csv.reader(f))  # reads the file and returns it, turning into list
# f = open("Football_players.csv") #opening file for reading
# f.close()

age_list = np.array([]) # we need empty numpy array to fill it
salary_list = np.array([])

# csv list is a nested list, so for that
for row in csv_list[1:]: # header kısmından dolayı error çıkmaması için header dahil edilmiyor
    age_list = np.append(age_list, int(row[4])) # rowlar int e çevrilmeli
    salary_list = np.append(salary_list, int(row[8]))
    # row[4]  reaching age column
    # row[8]  reaching salary column

plt.plot(age_list, salary_list, color="red", label="Line plot in red") # line plot
plt.scatter(age_list, salary_list,label="Scatterplot") # scatter plot
plt.xlabel("Age")
plt.ylabel("Salary")
# plt.legend(["Scatterplot", "Line plot in red"]) # we give list of labels
# plt.show() # call this at the very end

# plt.subplots() # bir ekranda çoklu gösterim
plt.figure() # creates a 2nd window
plt.plot(age_list, salary_list, color="red", label="Line plot in red") # line plot
plt.show()

print("Done.")
