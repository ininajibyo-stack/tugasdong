import pandas as pd
import numpy as np

# LOAD
kendaraan = pd.read_csv("kendaraan.csv")
jalan = pd.read_csv("jalan.csv")
penduduk = pd.read_csv("penduduk.csv")

# ========================
# KENDARAAN
# ========================
kendaraan["kendaraan"] = (
    kendaraan["mobil_penumpang"] +
    kendaraan["bus"] +
    kendaraan["truk"] +
    kendaraan["sepeda_motor"]
)

kendaraan = kendaraan[["nama_kabupaten_kota", "tahun", "kendaraan"]]

# ========================
# JALAN
# ========================
jalan = jalan.rename(columns={"panjang_ruas_jalan": "jalan"})
jalan = jalan[["nama_kabupaten_kota", "tahun", "jalan"]]

# ========================
# PENDUDUK
# ========================
penduduk = penduduk.rename(columns={"kepadatan_penduduk": "penduduk"})
penduduk = penduduk[["nama_kabupaten_kota", "tahun", "penduduk"]]

# ========================
# MERGE
# ========================
data = kendaraan.merge(jalan, on=["nama_kabupaten_kota", "tahun"])
data = data.merge(penduduk, on=["nama_kabupaten_kota", "tahun"])

# ========================
# 🔥 FIX INFINITY
# ========================
data = data[data["jalan"] > 0]

# ========================
# FITUR
# ========================
data["kepadatan"] = data["kendaraan"] / data["jalan"]

# HAPUS nilai aneh
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

# ========================
# LABEL (SEIMBANG)
# ========================
q1 = data["kepadatan"].quantile(0.33)
q2 = data["kepadatan"].quantile(0.66)

def label_macet(x):
    if x < q1:
        return 0  # Lancar
    elif x < q2:
        return 1  # Sedang
    else:
        return 2  # Macet

data["kemacetan"] = data["kepadatan"].apply(label_macet)

print(data["kemacetan"].value_counts())

# SAVE
data.to_csv("data_final.csv", index=False)

print("✅ Data bersih & siap!")
