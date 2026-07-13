import json
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

CITY_CONFIG = {
    "Bali": {"lat": -8.6705, "lon": 115.2126, "tz": "Asia/Makassar"},
    "Makassar": {"lat": -5.0611, "lon": 119.5540, "tz": "Asia/Makassar"},
    "Padang": {"lat": -0.9471, "lon": 100.4172, "tz": "Asia/Jakarta"},
    "Surabaya": {"lat": -7.2575, "lon": 112.7521, "tz": "Asia/Jakarta"},
}

SUPPORTED_CITIES = set(CITY_CONFIG.keys())


def wmo_to_deskripsi(code: int) -> str:
    """Map WMO weather code ke label yang dikenali model."""
    if code == 0:
        return "cerah"
    if code in (1, 2, 3):
        return "berawan"
    if code in (45, 48):
        return "berkabut"
    if code in (51, 53, 55, 56, 57):
        return "gerimis"
    if code in (61, 63, 65, 66, 67, 71, 73, 75, 77, 80, 81, 82, 85, 86):
        return "hujan ringan"
    if code in (95, 96, 99):
        return "badai petir"
    return "berawan"


def get_weather_for_city(city: str) -> dict:
    """Ambil cuaca terkini dari Open-Meteo untuk kota tujuan."""
    if city not in CITY_CONFIG:
        raise ValueError(f"Kota '{city}' tidak didukung")

    cfg = CITY_CONFIG[city]
    params = urlencode({
        "latitude": cfg["lat"],
        "longitude": cfg["lon"],
        "current": "temperature_2m,pressure_msl,wind_speed_10m,weather_code",
        "wind_speed_unit": "kmh",
        "timezone": cfg["tz"],
    })
    url = f"https://api.open-meteo.com/v1/forecast?{params}"

    try:
        with urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except URLError as exc:
        raise ConnectionError("Gagal mengambil data cuaca") from exc

    current = data.get("current", {})
    wmo_code = int(current.get("weather_code", 1))

    waktu_lokal = current.get("time", "")
    if "T" in waktu_lokal:
        waktu_lokal = waktu_lokal.split("T")[1][:5]

    return {
        "kota": city,
        "deskripsi_cuaca": wmo_to_deskripsi(wmo_code),
        "suhu": round(float(current.get("temperature_2m", 28)), 1),
        "tekanan": round(float(current.get("pressure_msl", 1013)), 1),
        "kecepatan_angin": round(float(current.get("wind_speed_10m", 10)), 1),
        "waktu_lokal": waktu_lokal,
        "zona_waktu": cfg["tz"],
    }
