# Import library
import numpy as np
from sklearn.linear_model import LinearRegression

# Data: misalnya kita punya data lama belajar (jam) dan hasil nilai ujian
# X adalah input (jumlah jam belajar), y adalah output (nilai ujian)
X = np.array([[1], [2], [3], [4], [5]])  # Jumlah jam belajar
y = np.array([50, 55, 65, 70, 80])      # Nilai ujian

# Membuat model Linear Regression
model = LinearRegression()

# Melatih model menggunakan data
model.fit(X, y)

# Prediksi nilai berdasarkan jam belajar
jam_belajar = np.array([[7]])  # Prediksi untuk 7 jam belajar
prediksi_nilai = model.predict(jam_belajar)

print(f"Prediksi nilai untuk 7 jam belajar: {prediksi_nilai[0]:.2f}")

from sklearn.metrics import mean_squared_error

# Prediksi untuk data training
y_pred = model.predict(X)

# Hitung Mean Squared Error (MSE)
mse = mean_squared_error(y, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
