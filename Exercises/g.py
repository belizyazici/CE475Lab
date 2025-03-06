import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Rastgele X verileri oluştur
np.random.seed(42)  # Sonuçları sabitlemek için
X = np.random.rand(100) * 10  # 0 ile 10 arasında 100 rastgele değer
Y = 2 * X + 5 + np.random.randn(100)  # Y = 2X + 5 + gürültü ekleyerek oluştur

# 2️⃣ Doğrusal regresyon hesaplama (Kapalı form çözümü)
N = len(X)
sum_XY = np.sum(X * Y)
sum_X = np.sum(X)
sum_Y = np.sum(Y)
sum_X2 = np.sum(X ** 2)

m = (N * sum_XY - sum_X * sum_Y) / (N * sum_X2 - sum_X ** 2)
b = (sum_Y - m * sum_X) / N

print(f"🔹 Hesaplanan Katsayılar: m = {m:.2f}, b = {b:.2f}")

# 3️⃣ Regresyon doğrusunu oluştur
X_line = np.linspace(0, 10, 100)
Y_line = m * X_line + b

# 4️⃣ Loss (MSE) Hesaplama
Y_pred = m * X + b  # Tahmin edilen Y değerleri
loss = np.mean((Y - Y_pred) ** 2)  # Mean Squared Error (MSE)
print(f"🔹 Modelin Loss (MSE) Değeri: {loss:.2f}")

# 5️⃣ Grafiği çizdir
plt.scatter(X, Y, color='blue', alpha=0.6, label="Veri Noktaları")  # Orijinal veriler
plt.plot(X_line, Y_line, color='red', label="Regresyon Doğrusu")  # Regresyon çizgisi
plt.xlabel("X Değeri")
plt.ylabel("Y Değeri")
plt.title("Basit Doğrusal Regresyon")
plt.legend()
plt.show()