from flask import Flask, request, render_template
import os
import joblib
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

# --------- Path aman relatif ke file ini ---------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Hasil_Uji_Data.csv")
PKL_PATH = os.path.join(BASE_DIR, "model_dan_encoders.joblib")

# --------- Muat dataset & model/encoders ---------
df_uji = pd.read_csv(CSV_PATH)

package = joblib.load(PKL_PATH)
model = package["model"]
label_encoders = package["label_encoders"]

le_maskapai = label_encoders["Maskapai"]
le_rute = label_encoders["Rute"]
le_deskripsi = label_encoders["Deskripsi_tujuan"]

# --------- Durasi normal per rute (menit) ---------
normal_duration_map = {
    "Jakarta-Padang": 110,
    "Jakarta-Surabaya": 80,
    "Jakarta-Denpasar": 170,
    "Jakarta-Makassar": 200,
}

print("✅ Model dan data berhasil dimuat")
print(f"📊 Dataset: {len(df_uji)} records")
print(f"🎯 Model: {type(model).__name__}")
print(f"📋 Rute tersedia: {list(normal_duration_map.keys())}")

# --------- Helpers ---------
def _dropdown_from_mapping():
    """Bangun asal/tujuan dari normal_duration_map dan hilangkan 'Jakarta' dari tujuan."""
    routes = sorted(list(normal_duration_map.keys()))
    asal_set, tujuan_set = set(), set()
    for r in routes:
        a, t = r.split("-")
        asal_set.add(a)
        tujuan_set.add(t)
    # Hilangkan Jakarta dari tujuan
    tujuan_set.discard("Jakarta")
    return {
        "asal_list": sorted(asal_set),
        "tujuan_list": sorted(tujuan_set),
        "available_routes": routes,
    }

@app.route("/")
def home():
    total_penerbangan = len(df_uji)
    daftar_maskapai = df_uji["Maskapai"].unique()
    daftar_rute = df_uji["Rute"].unique()
    bandara_asal = df_uji["Bandara_Asal"].unique()
    bandara_tujuan = df_uji["Bandara_Tujuan"].unique()

    info = {
        "total_penerbangan": total_penerbangan,
        "rute_maskapai": len(daftar_rute),
        "nama_maskapai": len(daftar_maskapai),
        "bandara_asal": len(bandara_asal),
        "bandara_tujuan": len(bandara_tujuan),
    }
    return render_template("index.html", info=info)

@app.route("/prediksi", methods=["GET", "POST"])
def prediksi():
    if request.method == "GET":
        # Dropdown yang pasti valid:
        maskapai_list = sorted(list(le_maskapai.classes_))
        cuaca_list = sorted(list(le_deskripsi.classes_))

        dd = _dropdown_from_mapping()
        return render_template(
            "prediksi.html",
            maskapai_list=maskapai_list,
            asal_list=dd["asal_list"],
            tujuan_list=dd["tujuan_list"],
            cuaca_list=cuaca_list,
            available_routes=dd["available_routes"],
            prediction_result=None,
            debug_info=None,
        )

    # ---------- POST ----------
    try:
        print("\n🔄 MEMULAI PREDIKSI")
        # Ambil data dari form
        tanggal_penerbangan = request.form["tanggal_penerbangan"]
        maskapai = request.form["maskapai"]
        asal = request.form["asal"]
        tujuan = request.form["tujuan"]
        deskripsi_cuaca = request.form["deskripsi_cuaca"]
        suhu = float(request.form["suhu"])
        tekanan = float(request.form["tekanan"])
        kecepatan_angin = float(request.form["kecepatan_angin"])
        jam_keberangkatan = request.form["jam_keberangkatan"]

        print(f"📅 Input: {tanggal_penerbangan} {jam_keberangkatan}")
        print(f"✈️ {maskapai}: {asal} → {tujuan}")

        # Buat rute dan validasi terhadap mapping
        rute = f"{asal}-{tujuan}"
        print(f"🗺️ Rute: {rute}")

        dd = _dropdown_from_mapping()  # untuk konsistensi dropdown di render apa pun cabangnya

        if rute not in normal_duration_map:
            prediction_result = (
                f"⚠️ Maaf, rute {rute} belum didukung. "
                f"Rute tersedia: {', '.join(normal_duration_map.keys())}"
            )
            return render_template(
                "prediksi.html",
                maskapai_list=sorted(list(le_maskapai.classes_)),
                asal_list=dd["asal_list"],
                tujuan_list=dd["tujuan_list"],
                cuaca_list=sorted(list(le_deskripsi.classes_)),
                available_routes=dd["available_routes"],
                prediction_result=prediction_result,
                debug_info=None,
            )

        durasi_normal = normal_duration_map[rute]
        print(f"⏱️ Durasi normal: {durasi_normal} menit (dari mapping)")

        # Convert tanggal ke ordinal
        tanggal_ordinal = pd.to_datetime(tanggal_penerbangan).toordinal()
        print(f"📅 Tanggal ordinal: {tanggal_ordinal}")

        # Encoding
        try:
            maskapai_enc = le_maskapai.transform([maskapai])[0]
            print(f"✈️ Maskapai encoded: {maskapai} → {maskapai_enc}")
        except Exception:
            prediction_result = f"⚠️ Maaf, maskapai {maskapai} belum dikenali."
            return render_template(
                "prediksi.html",
                maskapai_list=sorted(list(le_maskapai.classes_)),
                asal_list=dd["asal_list"],
                tujuan_list=dd["tujuan_list"],
                cuaca_list=sorted(list(le_deskripsi.classes_)),
                available_routes=dd["available_routes"],
                prediction_result=prediction_result,
                debug_info=None,
            )

        try:
            rute_enc = le_rute.transform([rute])[0]
            print(f"🗺️ Rute encoded: {rute} → {rute_enc}")
        except Exception:
            prediction_result = f"⚠️ Maaf, rute {rute} belum dikenali oleh encoder."
            return render_template(
                "prediksi.html",
                maskapai_list=sorted(list(le_maskapai.classes_)),
                asal_list=dd["asal_list"],
                tujuan_list=dd["tujuan_list"],
                cuaca_list=sorted(list(le_deskripsi.classes_)),
                available_routes=dd["available_routes"],
                prediction_result=prediction_result,
                debug_info=None,
            )

        try:
            deskripsi_enc = le_deskripsi.transform([deskripsi_cuaca])[0]
            print(f"🌤️ Cuaca encoded: {deskripsi_cuaca} → {deskripsi_enc}")
        except Exception:
            prediction_result = f"⚠️ Maaf, deskripsi cuaca '{deskripsi_cuaca}' belum dikenali."
            return render_template(
                "prediksi.html",
                maskapai_list=sorted(list(le_maskapai.classes_)),
                asal_list=dd["asal_list"],
                tujuan_list=dd["tujuan_list"],
                cuaca_list=sorted(list(le_deskripsi.classes_)),
                available_routes=dd["available_routes"],
                prediction_result=prediction_result,
                debug_info=None,
            )

        # Prediksi
        X = [[tanggal_ordinal, maskapai_enc, rute_enc, deskripsi_enc, suhu, tekanan, kecepatan_angin]]
        print(f"📊 Input untuk model: {X[0]}")
        prediksi_keterlambatan = model.predict(X)[0]
        print(f"🤖 Predicted delay: {prediksi_keterlambatan:.2f} menit")

        # Hitung ETA
        keberangkatan_dt = datetime.strptime(
            f"{tanggal_penerbangan} {jam_keberangkatan}", "%Y-%m-%d %H:%M"
        )
        jadwal_kedatangan_normal = keberangkatan_dt + timedelta(minutes=durasi_normal)
        menit_prediksi = durasi_normal + prediksi_keterlambatan
        waktu_prediksi_kedatangan = keberangkatan_dt + timedelta(minutes=menit_prediksi)

        print(f"🕐 Keberangkatan: {keberangkatan_dt.strftime('%H:%M')}")
        print(f"🕐 Jadwal kedatangan normal: {jadwal_kedatangan_normal.strftime('%H:%M:%S')}")
        print(f"🕐 Prediksi kedatangan final: {waktu_prediksi_kedatangan.strftime('%H:%M:%S')}")

        # Format hasil
        if prediksi_keterlambatan > 1:
            prediction_result = (
                f"{maskapai} diprediksi mengalami keterlambatan kedatangan sekitar "
                f"{int(prediksi_keterlambatan)} menit yaitu pukul {waktu_prediksi_kedatangan.strftime('%H:%M')}"
            )
        else:
            prediction_result = (
                f"➡️ {maskapai} diprediksi kedatangan tepat waktu yaitu pukul "
                f"{waktu_prediksi_kedatangan.strftime('%H:%M')}"
            )

        print(f"✅ HASIL: {prediction_result}")

        debug_info = {
            "maskapai": maskapai,
            "rute": rute,
            "tanggal_penerbangan": tanggal_penerbangan,
            "jam_keberangkatan": jam_keberangkatan,
            "bandara_asal": asal,
            "bandara_tujuan": tujuan,
            "durasi_normal": durasi_normal,
            "jadwal_kedatangan_normal": jadwal_kedatangan_normal.strftime("%H:%M"),
            "predicted_delay": round(prediksi_keterlambatan, 2),
            "prediksi_waktu_tiba": waktu_prediksi_kedatangan.strftime("%H:%M"),
            "data_source": "Normal Duration Mapping",
            "model_type": type(model).__name__,
            "input_features": {
                "tanggal_ordinal": tanggal_ordinal,
                "maskapai_encoded": f"{maskapai} → {maskapai_enc}",
                "rute_encoded": f"{rute} → {rute_enc}",
                "deskripsi_encoded": f"{deskripsi_cuaca} → {deskripsi_enc}",
                "suhu": suhu,
                "tekanan": tekanan,
                "kecepatan_angin": kecepatan_angin,
            },
        }

        return render_template(
            "prediksi.html",
            maskapai_list=sorted(list(le_maskapai.classes_)),
            asal_list=dd["asal_list"],
            tujuan_list=dd["tujuan_list"],
            cuaca_list=sorted(list(le_deskripsi.classes_)),
            available_routes=dd["available_routes"],
            prediction_result=prediction_result,
            debug_info=debug_info,
        )

    except Exception as e:
        print(f"💥 ERROR: {e}")
        import traceback
        traceback.print_exc()

        dd = _dropdown_from_mapping()
        prediction_result = f"⚠️ Terjadi kesalahan: {e}"
        return render_template(
            "prediksi.html",
            maskapai_list=sorted(list(le_maskapai.classes_)),
            asal_list=dd["asal_list"],
            tujuan_list=dd["tujuan_list"],
            cuaca_list=sorted(list(le_deskripsi.classes_)),
            available_routes=dd["available_routes"],
            prediction_result=prediction_result,
            debug_info=None,
        )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Alias untuk /prediksi"""
    return prediksi()

if __name__ == "__main__":
    print("🚀 Starting Flight Delay Prediction Server...")
    print("📍 Server: http://0.0.0.0:$PORT")
    print("🌐 Local: http://localhost:5000")
    print("📋 Routes: /prediksi, /predict")
    print("=" * 50)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
