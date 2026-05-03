import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# LOAD DATA
data = pd.read_csv("data_final.csv")

# FITUR
data["kepadatan"] = data["kendaraan"] / data["jalan"]

# HANDLE ERROR
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# LABEL OTOMATIS (3 KELAS)
q1 = data["kepadatan"].quantile(0.33)
q2 = data["kepadatan"].quantile(0.66)

def label(x):
    if x < q1:
        return 0
    elif x < q2:
        return 1
    else:
        return 2

data["kemacetan"] = data["kepadatan"].apply(label)

print("Distribusi Label:")
print(data["kemacetan"].value_counts())

# DATASET
X = data[["kepadatan", "penduduk"]]
y = to_categorical(data["kemacetan"], 3)

# SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# MODEL BACKPROPAGATION
model = Sequential([
    Dense(16, activation='relu', input_shape=(2,)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# TRAIN
history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test)
)

# SAVE MODEL
model.save("model.h5")

# BUAT FOLDER STATIC
if not os.path.exists("static"):
    os.makedirs("static")

# GRAFIK LOSS
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.savefig("static/loss.png")
plt.close()

# GRAFIK ACCURACY
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.savefig("static/accuracy.png")
plt.close()

print("✅ Model & grafik siap!")
