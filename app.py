from flask import Flask, render_template, request
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

app = Flask(__name__)

labels = ["Lancar", "Sedang", "Macet"]

# =========================
# LOAD ATAU TRAIN MODEL
# =========================
if os.path.exists("model.h5") and os.path.exists("scaler.pkl"):
    print("✅ Load model dari file")
    model = load_model("model.h5")
    scaler = joblib.load("scaler.pkl")
else:
    print("⚠️ Model tidak ditemukan, training ulang...")

    # DATA LATIH (LEBIH MASUK AKAL)
    X = np.array([
        [5, 500],
        [20, 1000],
        [50, 2000],
        [100, 3000],
        [200, 4000],
        [300, 6000],
        [500, 8000]
    ])

    y = np.array([0, 0, 1, 1, 2, 2, 2])  # 0=lancar,1=sedang,2=macet

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # MODEL BACKPROPAGATION
    model = Sequential([
        Dense(16, activation='relu', input_shape=(2,)),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_scaled, y, epochs=150, verbose=0)

    # SIMPAN MODEL (WAJIB)
    model.save("model.h5")
    joblib.dump(scaler, "scaler.pkl")

    print("✅ Model berhasil dibuat & disimpan")

# =========================
# ROUTE
# =========================
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    kendaraan = float(request.form['kendaraan'])
    jalan = float(request.form['jalan'])
    penduduk = float(request.form['penduduk'])

    if jalan == 0:
        return "❌ Jalan tidak boleh 0"

    kepadatan = kendaraan / jalan

    data = np.array([[kepadatan, penduduk]])
    data = scaler.transform(data)

    pred = model.predict(data)
    hasil_model = np.argmax(pred)

    # =========================
    # LOGIKA TAMBAHAN BIAR REALISTIS
    # =========================
    if kepadatan < 50:
        hasil_logika = 0
    elif kepadatan < 200:
        hasil_logika = 1
    else:
        hasil_logika = 2

    hasil = labels[hasil_model]

    # Sinkronisasi model + logika
    if abs(hasil_model - hasil_logika) > 1:
        hasil = labels[hasil_logika]

    return render_template(
        "result.html",
        hasil=hasil,
        kepadatan=round(kepadatan, 2)
    )

# =========================
if __name__ == "__main__":
    app.run(debug=True)
