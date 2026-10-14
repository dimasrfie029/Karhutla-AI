"""
🔥 Karhutla Risk AI – Explainable Forest Fire Risk Prediction for Selected Indonesian Location
Jalankan:  streamlit run app.py
Pastikan file logo.png berada di folder yang sama dengan app.py
"""
import hashlib
from datetime import datetime
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
APP_NAME = "Karhutla Risk AI"
APP_TAGLINE = "Explainable Forest Fire Risk Prediction for Selected Indonesian Location"
APP_TITLE = f"{APP_NAME} - {APP_TAGLINE}"

LOGO_PATH = Path(__file__).parent / "logo.png"


def _page_icon():
    if LOGO_PATH.exists():
        try:
            from PIL import Image
            return Image.open(LOGO_PATH)
        except Exception:
            pass
    return "🔥"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=_page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)


def show_logo(container=st, **kwargs):
    """Tampilkan logo.png bila ada (tidak error kalau file hilang)."""
    if LOGO_PATH.exists():
        container.image(str(LOGO_PATH), **kwargs)


# nama: (lat, lon, sebaran, bobot jumlah hotspot demo)
REGIONS = {
    "Riau": (0.50, 101.80, 0.6, 0.12),
    "Jambi": (-1.60, 103.00, 0.5, 0.07),
    "Sumatera Selatan": (-3.20, 104.20, 0.6, 0.10),
    "Sumatera Utara": (2.00, 99.00, 0.6, 0.04),
    "Kalimantan Barat": (-0.05, 110.60, 1.0, 0.12),
    "Kalimantan Tengah": (-1.70, 113.90, 1.0, 0.13),
    "Kalimantan Selatan": (-3.10, 115.30, 0.5, 0.06),
    "Kalimantan Timur": (0.50, 116.40, 1.0, 0.07),
    "Kalimantan Utara": (3.00, 116.60, 0.6, 0.03),
    "Jawa": (-7.30, 110.50, 0.8, 0.04),
    "Nusa Tenggara": (-9.00, 120.00, 1.0, 0.05),
    "Sulawesi": (-2.00, 121.00, 1.0, 0.06),
    "Maluku": (-3.30, 128.00, 1.0, 0.03),
    "Papua": (-5.50, 139.00, 1.0, 0.08),
}
INDONESIA_CENTER = (-2.5, 118.0)
# Batas wilayah Indonesia (lat_min, lat_max, lon_min, lon_max) untuk validasi unggahan
INDONESIA_BOUNDS = (-11.5, 6.5, 94.5, 141.5)

# Semua sumber tile di bawah gratis TANPA API key (CARTO kini mewajibkan key -> muncul watermark).
OSM_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_ATTR = "&copy; OpenStreetMap contributors"
TILES = {
    "Gelap (OSM)": {"url": OSM_URL, "attr": OSM_ATTR, "dark": True},
    "Terang (OSM)": {"url": OSM_URL, "attr": OSM_ATTR, "dark": False},
    "Satelit (Esri)": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri, Maxar, Earthstar Geographics", "dark": False,
    },
    "Topografi (Esri)": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri, HERE, Garmin, USGS", "dark": False,
    },
}

RISK_COLORS = {"RENDAH": "#2e7d32", "SEDANG": "#f9a825", "TINGGI": "#ef6c00", "EKSTREM": "#c62828"}


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


# Akun demo – GANTI sebelum dipakai sungguhan (idealnya simpan di st.secrets / database)
DEFAULT_USERS = {
    "admin": {"nama": "Admin Karhutla", "role": "Administrator", "pw": _hash("admin123")},
    "operator": {"nama": "Operator Lapangan", "role": "Operator", "pw": _hash("operator123")},
}

CSS = """
<style>
.block-container {padding-top: 2rem;}
div.stButton > button[kind="primary"] {
    background:#ff4b4b;border:none;width:100%;font-weight:600;
}
.danger-banner {
    background:linear-gradient(90deg,#7f1d1d,#b91c1c);color:white;padding:.6rem 1rem;
    border-radius:.5rem;font-weight:700;letter-spacing:.08em;text-align:center;margin-bottom:1rem;
}
.risk-pill {display:inline-block;padding:.15rem .7rem;border-radius:1rem;color:white;font-weight:700;}
.profile-card {background:#1a1d27;border-radius:.8rem;padding:1.2rem;border:1px solid #2b3040;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------------
def init_state():
    defaults = {
        "auth": False,
        "user": None,
        "users": {k: dict(v) for k, v in DEFAULT_USERS.items()},
        "logs": [],
        "pred_result": None,
        "data_source": "Data demo (simulasi)",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def add_log(pesan: str):
    st.session_state.logs.insert(
        0,
        {"waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         "user": st.session_state.get("user") or "-",
         "aktivitas": pesan},
    )
    del st.session_state.logs[200:]


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def generate_demo_hotspots(hour_key: str, n: int = 120) -> pd.DataFrame:
    """Data simulasi. Ganti dengan data NASA FIRMS / BMKG untuk produksi."""
    rng = np.random.default_rng(int(hour_key))
    names = list(REGIONS)
    weights = np.array([REGIONS[r][3] for r in names])
    chosen = rng.choice(names, size=n, p=weights / weights.sum())
    rows = []
    for i, reg in enumerate(chosen, 1):
        lat0, lon0, spread, _ = REGIONS[reg]
        rows.append({
            "id": f"HS-{i:04d}",
            "lokasi": reg,
            "lat": lat0 + rng.normal(0, 0.45 * spread),
            "lon": lon0 + rng.normal(0, 0.55 * spread),
            "suhu_k": round(float(rng.normal(345, 22)), 1),
            "confidence": int(np.clip(rng.normal(65, 22), 10, 100)),
            "umur_jam": int(rng.integers(1, 48)),
        })
    return pd.DataFrame(rows)


def assign_region(lat: np.ndarray, lon: np.ndarray) -> list:
    names = list(REGIONS)
    centers = np.array([[REGIONS[r][0], REGIONS[r][1]] for r in names])
    d = (lat[:, None] - centers[:, 0]) ** 2 + (lon[:, None] - centers[:, 1]) ** 2
    return [names[i] for i in d.argmin(axis=1)]


def normalize_upload(raw: pd.DataFrame) -> pd.DataFrame:
    """Terima CSV format NASA FIRMS (latitude, longitude, brightness, confidence, acq_date)."""
    cols = {c.lower(): c for c in raw.columns}
    lat_c = cols.get("latitude") or cols.get("lat")
    lon_c = cols.get("longitude") or cols.get("lon")
    if not lat_c or not lon_c:
        raise ValueError("Kolom latitude/longitude tidak ditemukan.")
    out = pd.DataFrame({
        "lat": pd.to_numeric(raw[lat_c], errors="coerce"),
        "lon": pd.to_numeric(raw[lon_c], errors="coerce"),
    })
    bright = cols.get("brightness") or cols.get("bright_ti4")
    out["suhu_k"] = pd.to_numeric(raw[bright], errors="coerce") if bright else 330.0
    if "confidence" in cols:
        conf = raw[cols["confidence"]]
        mapped = conf.astype(str).str.lower().map({"l": 30, "low": 30, "n": 60, "nominal": 60, "h": 90, "high": 90})
        out["confidence"] = mapped.fillna(pd.to_numeric(conf, errors="coerce")).fillna(50)
    else:
        out["confidence"] = 50
    if "acq_date" in cols:
        dt = pd.to_datetime(raw[cols["acq_date"]], errors="coerce")
        umur = (datetime.now() - dt).dt.total_seconds() / 3600
        out["umur_jam"] = umur.clip(lower=0).fillna(0).round().astype(int)
    else:
        out["umur_jam"] = 0
    lat_min, lat_max, lon_min, lon_max = INDONESIA_BOUNDS
    out = out.dropna(subset=["lat", "lon"])
    out = out[out.lat.between(lat_min, lat_max) & out.lon.between(lon_min, lon_max)].reset_index(drop=True)
    if out.empty:
        raise ValueError("Tidak ada titik di wilayah Indonesia pada file ini.")
    out["lokasi"] = assign_region(out.lat.to_numpy(), out.lon.to_numpy())
    out["confidence"] = out.confidence.astype(int)
    out["suhu_k"] = out.suhu_k.round(1)
    out.insert(0, "id", [f"HS-{i:04d}" for i in range(1, len(out) + 1)])
    return out


def get_hotspots() -> pd.DataFrame:
    if "hotspots" not in st.session_state:
        st.session_state.hotspots = generate_demo_hotspots(datetime.now().strftime("%Y%m%d%H"))
    return st.session_state.hotspots


@st.cache_data(ttl=900, show_spinner=False)
def fetch_weather(lat: float, lon: float):
    """Cuaca real-time gratis dari Open-Meteo (tanpa API key)."""
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon, "timezone": "auto",
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            },
            timeout=6,
        )
        r.raise_for_status()
        c = r.json()["current"]
        return {"temp": float(c["temperature_2m"]), "hum": float(c["relative_humidity_2m"]),
                "wind": float(c["wind_speed_10m"])}
    except Exception:
        return None


# ----------------------------------------------------------------------------
# MODEL RISIKO (heuristik transparan / explainable – bisa diganti model ML)
# ----------------------------------------------------------------------------
def risk_components(temp: float, hum: float, wind: float, recent_hotspots: int) -> dict:
    """Kontribusi tiap faktor terhadap skor (poin 0-100). Total = skor risiko."""
    t = np.clip((temp - 25) / 15, 0, 1)        # makin panas makin berisiko
    h = np.clip((80 - hum) / 60, 0, 1)         # makin kering makin berisiko
    w = np.clip(wind / 40, 0, 1)               # angin mempercepat penyebaran
    d = np.clip(recent_hotspots / 15, 0, 1)    # kepadatan hotspot 24 jam terakhir
    return {
        "Suhu": float(0.30 * t * 100),
        "Kekeringan (RH rendah)": float(0.30 * h * 100),
        "Angin": float(0.15 * w * 100),
        "Kepadatan hotspot": float(0.25 * d * 100),
    }


def risk_score(temp: float, hum: float, wind: float, recent_hotspots: int) -> float:
    return round(sum(risk_components(temp, hum, wind, recent_hotspots).values()), 1)


def risk_label(score: float) -> str:
    if score < 25:
        return "RENDAH"
    if score < 45:
        return "SEDANG"
    if score < 70:
        return "TINGGI"
    return "EKSTREM"


def pill(label: str) -> str:
    return f'<span class="risk-pill" style="background:{RISK_COLORS[label]}">{label}</span>'


def predict_24h(df: pd.DataFrame, weather: dict) -> pd.DataFrame:
    rows = []
    for name, (lat, lon, _, _) in REGIONS.items():
        recent = int(((df.lokasi == name) & (df.umur_jam <= 24)).sum())
        score = risk_score(weather["temp"], weather["hum"], weather["wind"], recent)
        rows.append({
            "Wilayah": name, "lat": lat, "lon": lon,
            "Hotspot 24j Terakhir": recent,
            "Skor Risiko": score,
            "Level": risk_label(score),
            "Prediksi Hotspot 24j": int(round(recent * (1 + score / 100))),
        })
    return pd.DataFrame(rows).sort_values("Skor Risiko", ascending=False).reset_index(drop=True)


def filter_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    return df if region == "Semua" else df[df.lokasi == region]


def recent_for(df: pd.DataFrame, region: str) -> int:
    """Jumlah hotspot 24 jam. Untuk 'Semua' dipakai rata-rata per wilayah agar skala sebanding."""
    recent = df[df.umur_jam <= 24]
    if region == "Semua":
        return int(round(len(recent) / len(REGIONS)))
    return int((recent.lokasi == region).sum())


# ----------------------------------------------------------------------------
# PETA
# ----------------------------------------------------------------------------
def build_map(df, pred, tile, region, show_points, show_heat, show_pred, cluster):
    center, zoom = INDONESIA_CENTER, 5
    if region != "Semua":
        center, zoom = (REGIONS[region][0], REGIONS[region][1]), 7

    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True,
                   min_zoom=4, max_zoom=17)
    folium.TileLayer(
        tiles=tile["url"], attr=tile["attr"], name="Peta dasar",
        className="dark-tiles" if tile["dark"] else "", max_zoom=17,
    ).add_to(m)
    # Mode gelap: peta OSM biasa dibalik warnanya lewat CSS filter.
    m.get_root().html.add_child(folium.Element(
        "<style>.dark-tiles{filter:invert(1) hue-rotate(180deg) brightness(.85) contrast(.9);}"
        ".leaflet-container{background:#1b1f2a;}</style>"))

    # Zoom otomatis ke sebaran titik supaya tidak menumpuk di satu tempat.
    if len(df):
        m.fit_bounds([[df.lat.min(), df.lon.min()], [df.lat.max(), df.lon.max()]],
                     padding=(30, 30), max_zoom=9)
    else:
        m.fit_bounds([[-11.0, 95.0], [6.0, 141.0]])  # seluruh Indonesia

    if show_heat and len(df):
        HeatMap(df[["lat", "lon", "confidence"]].values.tolist(), name="Heatmap", radius=22, blur=18,
                min_opacity=0.35).add_to(m)

    if show_points and len(df):
        group = folium.FeatureGroup(name="Titik Hotspot")
        target = MarkerCluster().add_to(group) if cluster else group
        for r in df.itertuples():
            color = "#ff1744" if r.confidence >= 80 else "#ff9100" if r.confidence >= 50 else "#ffd600"
            popup = (f"<b>{r.id}</b><br>{r.lokasi}<br>Suhu: {r.suhu_k} K<br>"
                     f"Confidence: {r.confidence}%<br>Umur: {r.umur_jam} jam")
            folium.CircleMarker(
                [r.lat, r.lon], radius=6, color=color, fill=True, fill_opacity=0.85,
                popup=folium.Popup(popup, max_width=220), tooltip=f"{r.id} • {r.confidence}%",
            ).add_to(target)
        group.add_to(m)

    if show_pred:
        group = folium.FeatureGroup(name="Zona Risiko 24 Jam")
        for r in pred.itertuples():
            skor = r._5  # kolom "Skor Risiko"
            folium.Circle(
                [r.lat, r.lon], radius=40000 + skor * 2500, color=RISK_COLORS[r.Level],
                fill=True, fill_opacity=0.18, weight=2,
                tooltip=f"{r.Wilayah}: {r.Level} ({skor})",
            ).add_to(group)
        group.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    return m


# ----------------------------------------------------------------------------
# LOGIN
# ----------------------------------------------------------------------------
def login_page():
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("<div class='danger-banner'>⚠️ DANGER – FIRE AREA</div>", unsafe_allow_html=True)
        _, c, _ = st.columns([1, 1, 1])
        show_logo(c, use_container_width=True)
        st.markdown(f"<h2 style='text-align:center;margin-bottom:0'>{APP_NAME}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;font-size:1.05rem'>{APP_TAGLINE}</p>", unsafe_allow_html=True)
        st.caption("Sistem monitoring titik panas & prediksi risiko kebakaran hutan dan lahan di Indonesia.")
        with st.form("login"):
            uid = st.text_input("User ID")
            pw = st.text_input("Password", type="password")
            ok = st.form_submit_button("MASUK", type="primary", use_container_width=True)
        if ok:
            u = st.session_state.users.get(uid.strip().lower())
            if u and u["pw"] == _hash(pw):
                st.session_state.auth = True
                st.session_state.user = uid.strip().lower()
                add_log("Login berhasil")
                st.rerun()
            else:
                st.error("User ID atau password salah.")
        with st.expander("Akun demo"):
            st.code("admin / admin123\noperator / operator123")
            st.caption("Ganti akun ini sebelum dipublikasikan.")


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
PAGES = ["🗺️ Peta Monitoring", "📋 Data Hotspot", "🤖 Prediksi AI", "👤 Profil & Admin", "❓ FAQ"]


def sidebar(df):
    with st.sidebar:
        show_logo(st, width=140)
        st.markdown(f"**{APP_NAME}**")
        page = st.radio("Navigasi", PAGES, label_visibility="collapsed")
        st.divider()

        region = st.selectbox("Wilayah", ["Semua"] + list(REGIONS))

        st.markdown("### 📊 Indikator Cuaca")
        mode = st.radio("Sumber", ["Otomatis (Open-Meteo)", "Manual"], horizontal=True,
                        label_visibility="collapsed")
        lat, lon = (INDONESIA_CENTER if region == "Semua" else REGIONS[region][:2])
        weather = None
        if mode.startswith("Otomatis"):
            weather = fetch_weather(round(lat, 2), round(lon, 2))
            if weather is None:
                st.warning("Gagal mengambil cuaca live, pakai nilai manual.")
        if weather is None:
            weather = {
                "temp": st.number_input("Suhu (°C)", 15.0, 50.0, 34.0, 0.5),
                "hum": st.number_input("Kelembapan (%)", 0.0, 100.0, 45.0, 1.0),
                "wind": st.number_input("Angin (km/h)", 0.0, 120.0, 18.0, 1.0),
            }
        else:
            st.caption(f"Live • {datetime.now():%H:%M}")
            if region == "Semua":
                st.caption("Pilih wilayah tertentu untuk cuaca yang lebih akurat.")

        st.metric("🌡️ Suhu", f"{weather['temp']:.0f}°C")
        st.metric("💧 Kelembapan", f"{weather['hum']:.0f}%")
        st.metric("🌬️ Angin", f"{weather['wind']:.0f} km/h")
        st.divider()

        st.markdown("### 🔮 AI Fire Predict in 24h")
        if st.button("Predict Now", type="primary"):
            recent = recent_for(df, region)
            score = risk_score(weather["temp"], weather["hum"], weather["wind"], recent)
            st.session_state.pred_result = {
                "score": score, "label": risk_label(score), "region": region,
                "time": datetime.now().strftime("%H:%M"),
            }
            add_log(f"Prediksi AI 24 jam ({region}) → {risk_label(score)} ({score})")
        res = st.session_state.pred_result
        if res:
            st.markdown(f"Risiko Terhitung: {pill(res['label'])}", unsafe_allow_html=True)
            st.caption(f"Skor {res['score']}/100 • {res['region']} • {res['time']}")
        else:
            st.caption("Belum ada prediksi. Klik **Predict Now**.")
        st.divider()

        u = st.session_state.users[st.session_state.user]
        st.caption(f"Masuk sebagai **{u['nama']}** ({u['role']})")
        if st.button("Keluar"):
            add_log("Logout")
            st.session_state.auth = False
            st.session_state.user = None
            st.rerun()
    return page, region, weather


# ----------------------------------------------------------------------------
# HALAMAN
# ----------------------------------------------------------------------------
def page_map(df, region, weather):
    st.title("🗺️ Map Monitoring: Hotspots Indonesia")
    st.caption(APP_TITLE)
    view = filter_region(df, region)

    with st.expander("⚙️ Layer & Jenis Peta", expanded=False):
        c1, c2 = st.columns([1.2, 2])
        tile_name = c1.selectbox("Jenis peta", list(TILES))
        min_conf = c1.slider("Confidence minimum (%)", 0, 100, 0, 5)
        a, b, c, d = c2.columns(4)
        show_points = a.checkbox("Titik hotspot", True)
        cluster = b.checkbox("Cluster", False)
        show_heat = c.checkbox("Heatmap", False)
        show_pred = d.checkbox("Zona risiko", True)
    view = view[view.confidence >= min_conf]
    pred = predict_24h(df, weather)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total hotspot", len(view))
    k2.metric("Baru (≤24 jam)", int((view.umur_jam <= 24).sum()))
    k3.metric("Confidence tinggi (≥80%)", int((view.confidence >= 80).sum()))
    top = pred.iloc[0]
    k4.metric("Wilayah risiko tertinggi", top["Wilayah"], top["Level"], delta_color="off")

    m = build_map(view, pred, TILES[tile_name], region, show_points, show_heat, show_pred, cluster)
    st_folium(m, height=540, use_container_width=True, returned_objects=[])

    recent = view[view.umur_jam <= 24]
    if len(recent):
        counts = recent.lokasi.value_counts()
        st.error(f"🚨 **ALERT:** {counts.iloc[0]} titik panas baru terdeteksi di wilayah "
                 f"{counts.index[0]}! Harap tingkatkan kewaspadaan.")
    else:
        st.success("✅ Tidak ada titik panas baru dalam 24 jam terakhir pada filter ini.")
    st.caption(f"Sumber data: {st.session_state.data_source}")


def page_data(df, region):
    st.title("📋 Data Hotspot")
    view = filter_region(df, region)

    c1, c2, c3 = st.columns([2, 1.5, 1.5])
    q = c1.text_input("🔍 Cari ID / lokasi")
    conf = c2.slider("Confidence minimum", 0, 100, 0, 5)
    umur = c3.slider("Umur maksimum (jam)", 1, 72, 48)
    view = view[(view.confidence >= conf) & (view.umur_jam <= umur)]
    if q:
        view = view[view.id.str.contains(q, case=False) | view.lokasi.str.contains(q, case=False)]

    st.dataframe(
        view.sort_values("umur_jam"),
        use_container_width=True, hide_index=True, height=360,
        column_config={
            "id": "ID", "lokasi": "Lokasi", "lat": st.column_config.NumberColumn("Lat", format="%.3f"),
            "lon": st.column_config.NumberColumn("Lon", format="%.3f"),
            "suhu_k": st.column_config.NumberColumn("Suhu (K)", format="%.1f"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%d%%"),
            "umur_jam": st.column_config.NumberColumn("Umur (jam)"),
        },
    )
    st.download_button("⬇️ Unduh CSV", view.to_csv(index=False).encode(), "hotspot_indonesia.csv", "text/csv")

    g1, g2 = st.columns(2)
    if len(view):
        by_loc = view.lokasi.value_counts().rename_axis("Wilayah").reset_index(name="Jumlah")
        g1.plotly_chart(px.bar(by_loc, x="Wilayah", y="Jumlah", title="Hotspot per wilayah",
                               template="plotly_dark", color="Jumlah", color_continuous_scale="Reds"),
                        use_container_width=True)
        g2.plotly_chart(px.histogram(view, x="umur_jam", nbins=16, title="Sebaran umur hotspot (jam)",
                                     template="plotly_dark", color_discrete_sequence=["#ff4b4b"]),
                        use_container_width=True)
    else:
        st.info("Tidak ada data yang cocok dengan filter.")

    with st.expander("📤 Unggah data NASA FIRMS (CSV)"):
        f = st.file_uploader("File CSV", type="csv")
        if f is not None:
            try:
                new = normalize_upload(pd.read_csv(f))
                st.success(f"{len(new)} titik valid di Indonesia.")
                if st.button("Gunakan data ini", type="primary"):
                    st.session_state.hotspots = new
                    st.session_state.data_source = f"Upload: {f.name}"
                    add_log(f"Upload data {f.name} ({len(new)} titik)")
                    st.rerun()
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")
        if st.session_state.data_source != "Data demo (simulasi)" and st.button("Kembali ke data demo"):
            st.session_state.pop("hotspots", None)
            st.session_state.data_source = "Data demo (simulasi)"
            st.rerun()


def page_predict(df, weather):
    st.title("🤖 AI Fire Predict in 24H")
    pick = st.selectbox("Wilayah", ["Semua"] + list(REGIONS))

    pred = predict_24h(df, weather)
    view = pred if pick == "Semua" else pred[pred.Wilayah == pick]
    recent = recent_for(df, pick)
    comps = risk_components(weather["temp"], weather["hum"], weather["wind"], recent)
    score = round(sum(comps.values()), 1)
    label = risk_label(score)

    left, right = st.columns([1.7, 1])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score, title={"text": f"Skor Risiko 24 Jam – {pick}"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": RISK_COLORS[label]},
                   "steps": [{"range": [0, 25], "color": "#1b3a26"}, {"range": [25, 45], "color": "#4a4620"},
                             {"range": [45, 70], "color": "#5a3a1a"}, {"range": [70, 100], "color": "#5a1a1a"}]},
        ))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", font={"color": "#fafafa"})
        st.plotly_chart(fig, use_container_width=True)

        # Penjelasan (explainable): kontribusi tiap faktor terhadap skor
        st.markdown("#### 🔍 Mengapa skornya segini? (kontribusi faktor)")
        exp = pd.DataFrame({"Faktor": list(comps), "Poin": [round(v, 1) for v in comps.values()]})
        st.plotly_chart(
            px.bar(exp, x="Poin", y="Faktor", orientation="h", template="plotly_dark",
                   text="Poin", color_discrete_sequence=["#ff4b4b"]),
            use_container_width=True,
        )

        st.markdown("#### Tabel prediksi per wilayah")
        st.dataframe(
            view[["Wilayah", "Hotspot 24j Terakhir", "Prediksi Hotspot 24j", "Skor Risiko", "Level"]],
            use_container_width=True, hide_index=True,
            column_config={"Skor Risiko": st.column_config.ProgressColumn("Skor Risiko", min_value=0,
                                                                          max_value=100, format="%.1f")},
        )
        st.plotly_chart(
            px.bar(pred, x="Wilayah", y="Skor Risiko", color="Level", color_discrete_map=RISK_COLORS,
                   template="plotly_dark", title="Perbandingan risiko antar wilayah"),
            use_container_width=True,
        )
        st.caption("Skor = 30% suhu + 30% kekeringan + 15% angin + 25% kepadatan hotspot 24 jam. "
                   "Ini model heuristik yang transparan; ganti dengan model ML untuk akurasi lebih baik.")

    with right:
        st.markdown("#### 🚨 Alert")
        danger = view[view.Level.isin(["TINGGI", "EKSTREM"])]
        if danger.empty:
            st.success("Tidak ada wilayah berisiko tinggi.")
        for r in danger.itertuples():
            (st.error if r.Level == "EKSTREM" else st.warning)(
                f"**{r.Wilayah}** – {r.Level} (skor {r._5})")
        with st.expander("📑 Data lengkap"):
            st.dataframe(view.drop(columns=["lat", "lon"]), use_container_width=True, hide_index=True)
        st.markdown("#### 🛠️ Quick tools")
        st.download_button("⬇️ Unduh prediksi (CSV)", pred.drop(columns=["lat", "lon"]).to_csv(index=False).encode(),
                           "prediksi_24j.csv", "text/csv", use_container_width=True)


def page_admin():
    st.title("👤 Profil & Data Admin")
    me = st.session_state.users[st.session_state.user]
    is_admin = me["role"] == "Administrator"
    tab1, tab2 = st.tabs(["Profil", "Data Admin"])

    with tab1:
        st.markdown(
            f"<div class='profile-card'><h2>👤 {me['nama']}</h2>"
            f"<p>{me['role']} • ID: <code>{st.session_state.user}</code></p></div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### Recent Reports")
        mine = [l for l in st.session_state.logs if l["user"] == st.session_state.user][:5]
        if mine:
            for l in mine:
                st.write(f"• `{l['waktu']}` – {l['aktivitas']}")
        else:
            st.caption("Belum ada aktivitas.")

    with tab2:
        if not is_admin:
            st.info("Bagian ini hanya untuk Administrator.")
            return
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### User Roles")
            st.dataframe(
                pd.DataFrame([{"User ID": k, "Nama": v["nama"], "Role": v["role"]}
                              for k, v in st.session_state.users.items()]),
                use_container_width=True, hide_index=True)
        with c2:
            st.markdown("#### System Logs")
            if st.session_state.logs:
                st.dataframe(pd.DataFrame(st.session_state.logs), use_container_width=True,
                             hide_index=True, height=215)
            else:
                st.caption("Log kosong.")

        st.markdown("#### Manage Users")
        a, b = st.columns(2)
        with a.form("add_user", clear_on_submit=True):
            st.write("**Tambah pengguna**")
            uid = st.text_input("User ID baru")
            nama = st.text_input("Nama lengkap")
            role = st.selectbox("Role", ["Operator", "Administrator"])
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Tambah", type="primary"):
                uid = uid.strip().lower()
                if not uid or not pw or not nama:
                    st.error("Semua kolom wajib diisi.")
                elif uid in st.session_state.users:
                    st.error("User ID sudah ada.")
                else:
                    st.session_state.users[uid] = {"nama": nama, "role": role, "pw": _hash(pw)}
                    add_log(f"Tambah user {uid} ({role})")
                    st.success(f"User {uid} ditambahkan.")
                    st.rerun()
        with b:
            st.write("**Hapus pengguna**")
            deletable = [u for u in st.session_state.users if u != st.session_state.user]
            target = st.selectbox("Pilih user", deletable) if deletable else None
            if target and st.button("Hapus user"):
                del st.session_state.users[target]
                add_log(f"Hapus user {target}")
                st.rerun()
        st.caption("Pengguna tersimpan di sesi saja. Untuk produksi gunakan database / st.secrets.")


def page_faq():
    st.title("❓ FAQ")
    faqs = {
        "Apa itu Karhutla Risk AI?": "Dashboard monitoring titik panas dan prediksi risiko kebakaran hutan dan lahan (karhutla) "
                                     "di beberapa wilayah Indonesia, dengan penjelasan kontribusi tiap faktor risiko.",
        "Apa itu hotspot?": "Titik panas hasil deteksi satelit (mis. MODIS/VIIRS) yang mengindikasikan kemungkinan kebakaran.",
        "Apa arti confidence?": "Tingkat keyakinan satelit bahwa titik tersebut benar-benar api. ≥80% dianggap tinggi.",
        "Bagaimana prediksi 24 jam dihitung?": "Gabungan suhu, kelembapan, kecepatan angin, dan jumlah hotspot 24 jam terakhir. "
                                               "Kontribusi tiap faktor ditampilkan di halaman Prediksi AI.",
        "Dari mana data cuaca?": "Open-Meteo (real-time). Anda bisa beralih ke input manual di sidebar.",
        "Bagaimana memakai data asli?": "Unduh CSV dari NASA FIRMS lalu unggah lewat halaman Data Hotspot.",
    }
    for q, a in faqs.items():
        with st.expander(q):
            st.write(a)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    init_state()
    if not st.session_state.auth:
        login_page()
        return

    df = get_hotspots()
    page, region, weather = sidebar(df)

    if page == PAGES[0]:
        page_map(df, region, weather)
    elif page == PAGES[1]:
        page_data(df, region)
    elif page == PAGES[2]:
        page_predict(df, weather)
    elif page == PAGES[3]:
        page_admin()
    else:
        page_faq()


main()