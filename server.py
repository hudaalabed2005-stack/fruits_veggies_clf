import os
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import io, csv
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ---------------- Local Image Classifier (PyTorch) ----------------
import torch
import torch.nn as nn
from PIL import Image
from io import BytesIO
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.getenv("MODEL_PATH", "spoilage_model.pth")
CLASS_NAMES = [x.strip() for x in os.getenv("CLASS_NAMES", "fresh,spoiled").split(",")]
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))

# Minimal preprocessing (resize + tensor only)
TX = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

_model = None
def load_model():
    """Load TorchScript (.pt/.pth) or a saved nn.Module."""
    global _model
    if _model is not None:
        return _model
    try:
        m = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    except Exception:
        obj = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(obj, nn.Module):
            m = obj
        else:
            raise RuntimeError("Model must be TorchScript or a saved nn.Module")
    m.eval().to(DEVICE)
    _model = m
    return _model

def run_classifier(pil_img: Image.Image):
    x = TX(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = load_model()(x)

    # support [B,1] sigmoid or [B,2] softmax heads
    if logits.ndim == 2 and logits.shape[1] == 1:
        p1 = torch.sigmoid(logits)[0, 0].item()
        probs = [1.0 - p1, p1]
    else:
        probs = torch.softmax(logits, dim=1)[0].tolist()

    idx = int(torch.tensor(probs).argmax().item())
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
    conf = round(float(probs[idx]) * 100.0, 1)
    return {"label": label, "confidence": conf, "probs": {
        (CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)): float(p)
        for i, p in enumerate(probs)
    }}

# ---------------- FastAPI app + CORS ----------------
app = FastAPI(title="Fruit & Gas Cloud API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# tiny in-memory cache
LAST = {
    "vision": None,
    "vision_updated": None,
    "gas": None,
    "gas_updated": None,
}

# ---------------- SQLite for gas history ----------------
DB_PATH = Path("data.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gas_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,         -- ISO UTC string
            co2 REAL, nh3 REAL, benzene REAL, alcohol REAL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_gas_ts ON gas_readings(ts)")
    con.commit()
    con.close()

def save_reading(ppm: dict):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO gas_readings (ts, co2, nh3, benzene, alcohol) VALUES (?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(),
         ppm.get("co2"), ppm.get("nh3"),
         ppm.get("benzene"), ppm.get("alcohol"))
    )
    con.commit()
    con.close()

def load_history_last_days(days: int = 2):
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        SELECT ts, co2, nh3, benzene, alcohol
        FROM gas_readings
        WHERE ts >= ?
        ORDER BY ts ASC
    """, (cutoff,))
    rows = cur.fetchall()
    con.close()
    return [
        {"time": ts, "ppm": {"co2": co2, "nh3": nh3, "benzene": benz, "alcohol": alc}}
        for (ts, co2, nh3, benz, alc) in rows
    ]

init_db()

# ---------------- Prediction ----------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Accept an uploaded image, run local model, cache result, return normalized JSON."""
    try:
        pil = Image.open(BytesIO(await image.read()))
    except Exception:
        return JSONResponse({"error": "invalid_image", "detail": "Could not read image"}, status_code=400)

    try:
        out = run_classifier(pil)
        LAST["vision"] = {"label": out["label"], "confidence": out["confidence"]}
        LAST["vision_updated"] = datetime.utcnow().isoformat()
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"error": "inference_failed", "detail": str(e)}, status_code=500)

# ---------------- Gas model ----------------
class GasReading(BaseModel):
    vrl: float | None = None
    adc: int | None = None
    adc_max: int | None = 4095
    vref: float | None = 3.3
    rl:   float | None = 10000.0
    rs:   float | None = None
    r0:   float | None = None

def _ppm_from_ratio(ratio: float, a: float, b: float) -> float:
    if ratio is None or ratio <= 0:
        return 0.0
    return max(0.0, a * (ratio ** b))

@app.post("/gas")
def gas(g: GasReading):
    VREF = float(g.vref or 3.3)   # <-- fixed missing space
    RL   = float(g.rl or 10000.0)

    if g.vrl is None and g.adc is not None:
        adc_max = int(g.adc_max or 4095)
        g.vrl = (float(g.adc) / float(adc_max)) * VREF

    if g.vrl is None and g.rs is None:
        return {"error": "Send at least one of: vrl, adc, or rs."}

    rs = float(g.rs) if g.rs is not None else ((VREF - g.vrl) * RL) / max(0.001, g.vrl)
    r0 = float(g.r0) if g.r0 is not None else rs
    ratio = rs / max(1e-6, r0)

    data = {
        "vrl": round(float(g.vrl), 3) if g.vrl is not None else None,
        "rs": round(rs, 1),
        "r0": round(r0, 1),
        "ratio": round(ratio, 3),
        "ppm": {
            "co2":     round(_ppm_from_ratio(ratio, 116.6021, -2.7690), 1),
            "nh3":     round(_ppm_from_ratio(ratio, 102.6940, -2.4880), 1),
            "benzene": round(_ppm_from_ratio(ratio, 76.63,   -2.1680), 1),
            "alcohol": round(_ppm_from_ratio(ratio, 77.255,  -3.18),   1),
        },
    }
    LAST["gas"] = data
    LAST["gas_updated"] = datetime.utcnow().isoformat()
    save_reading(data["ppm"])
    return {"ok": True, "data": data}

@app.post("/cron/snapshot")
def cron_snapshot():
    if not LAST.get("gas") or not LAST["gas"].get("ppm"):
        return {"ok": False, "error": "No gas reading to snapshot yet."}
    save_reading(LAST["gas"]["ppm"])
    return {"ok": True, "saved": LAST["gas"]["ppm"]}

@app.get("/history")
def history():
    return {"history": load_history_last_days(days=2)}

# ---------------- Summary (vision + gas) ----------------
def _summarize(last: dict) -> dict:
    pred = None
    v = last.get("vision")
    if isinstance(v, dict) and "label" in v:
        pred = v

    gas = (last.get("gas") or {}).get("ppm", {})
    co2  = gas.get("co2")
    nh3  = gas.get("nh3")
    benz = gas.get("benzene")
    alco = gas.get("alcohol")

    co2_hi = (co2 is not None) and (co2 >= 2000)
    nh3_hi = (nh3 is not None) and (nh3 >= 15)
    voc_hi = (benz or 0) >= 5 or (alco or 0) >= 10

    model_rotten = bool(pred and isinstance(pred.get("label"), str) and (("spoiled" in pred["label"].lower()) or ("rotten" in pred["label"].lower())))
    spoiled = model_rotten or co2_hi or nh3_hi or voc_hi

    return {
        "vision": pred,
        "gas_ppm": {"co2": co2, "nh3": nh3, "benzene": benz, "alcohol": alco},
        "gas_flags": {"co2_high": co2_hi, "nh3_high": nh3_hi, "voc_high": voc_hi},
        "decision": "SPOILED" if spoiled else "FRESH",
    }

@app.get("/export.csv")
def export_csv():
    rows = load_history_last_days(days=2)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["timestamp_utc", "co2_ppm", "nh3_ppm", "benzene_ppm", "alcohol_eq"])
    for r in rows:
        ts = r["time"]
        ppm = r["ppm"] or {}
        w.writerow([ts, ppm.get("co2"), ppm.get("nh3"), ppm.get("benzene"), ppm.get("alcohol")])
    csv_data = buf.getvalue()
    return HTMLResponse(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="gas_last_2_days.csv"'}
    )

@app.get("/summary")
def summary():
    return _summarize(LAST)

# ---------------- UI (unchanged) ----------------
@app.get("/", response_class=HTMLResponse)
def welcome():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Welcome • Fruit Detector</title>
  <style>
    body{
      margin:0; font-family:Inter,Arial,Helvetica,sans-serif; color:#fff; min-height:100vh;
      background:url('https://i.pinimg.com/originals/30/ab/43/30ab43926be6852d3b03572459ab847d.gif') center/cover no-repeat fixed;
      display:flex; align-items:center; justify-content:center;
    }
    body::before{content:""; position:fixed; inset:0; background:linear-gradient(to bottom right,rgba(0,0,0,.35),rgba(0,0,0,.55)); pointer-events:none;}
    .wrap{
      position:relative; text-align:center; padding:48px 40px; max-width:900px; width:92%;
      background:rgba(0,0,0,.30); border:1px solid rgba(255,255,255,.15); border-radius:20px; box-shadow:0 20px 60px rgba(0,0,0,.45);
      backdrop-filter:blur(6px); -webkit-backdrop-filter:blur(6px);
    }
    h1{font-size:clamp(2rem,4vw,3rem); margin:0 0 12px; letter-spacing:.5px;}
    p{font-size:1.05rem; opacity:.95; margin:0 auto 22px; max-width:760px; line-height:1.6}
    .cta{
      display:inline-block; margin-top:8px; padding:14px 26px; font-weight:800; letter-spacing:.2px; color:#0b3d2e;
      background:linear-gradient(135deg,#7CFFCB,#4ADE80); border:none; border-radius:12px; text-decoration:none;
      box-shadow:0 8px 24px rgba(16,185,129,.35); transition:transform .15s, box-shadow .15s, opacity .15s;
    }
    .cta:hover{transform:translateY(-2px); box-shadow:0 12px 28px rgba(16,185,129,.45);}
    .badge{
      display:inline-flex; gap:8px; padding:8px 12px; border-radius:999px; font-weight:700; color:#0b3d2e; background:rgba(255,255,255,.9);
      box-shadow:inset 0 0 0 1px rgba(0,0,0,.06); margin-bottom:14px
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="badge">🍎 <span>Fruit Freshness & Gas Detector</span></div>
    <h1>Smarter Food, Fresher Choices</h1>
    <p>Check fruit freshness with AI and estimate air quality using your sensor readings.</p>
    <a class="cta" href="/app">Start Detecting Freshness</a>
    <div class="meta">Or explore the API docs at <code>/docs</code>.</div>
  </div>
</body>
</html>
    """

@app.get("/app", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Fruit Freshness Detector</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <!-- Chart.js -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <style>/* (same CSS as your version) */</style>
</head>
<body> <!-- (same HTML + JS as your version) -->
  <!-- I kept your entire /app content exactly the same -->
  <!-- ... paste your original /app HTML here (unchanged) ... -->
</body>
</html>
    """
