from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

labels = ["Lancar", "Sedang", "Macet"]

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

    # LOGIKA TAMBAHAN (BIAR MASUK AKAL)
    if kepadatan < 50:
        hasil_logika = 0
    elif kepadatan < 200:
        hasil_logika = 1
    else:
        hasil_logika = 2

    hasil = labels[hasil_model]

    if abs(hasil_model - hasil_logika) > 1:
        hasil = labels[hasil_logika]

    return render_template(
        "result.html",
        hasil=hasil,
        kepadatan=round(kepadatan, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
