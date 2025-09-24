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

# ---------------- Roboflow ----------------
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
PROJECT = "fresh-or-rotten-detection-briat"
VERSION = "1"
CLASSIFY_URL = f"https://classify.roboflow.com/{PROJECT}/{VERSION}"

if not ROBOFLOW_API_KEY:
    raise RuntimeError("Missing ROBOFLOW_API_KEY environment variable.")

# ---------------- FastAPI app ----------------
app = FastAPI(title="Fruit & Gas Cloud API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Cache ----------------
LAST = {
    "vision": None,
    "vision_updated": None,
    "gas": None,
    "gas_updated": None,
}

# ---------------- SQLite ----------------
DB_PATH = Path("data.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gas_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
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

# ---------------- Vision helpers ----------------
def extract_top_class(resp_obj):
    if not resp_obj:
        return None
    preds = resp_obj.get("predictions")
    if preds is None:
        return None
    if isinstance(preds, list):
        preds_sorted = sorted(preds, key=lambda p: p.get("confidence", 0.0), reverse=True)
        if preds_sorted:
            return {
                "label": preds_sorted[0].get("class", "?"),
                "confidence": round(float(preds_sorted[0].get("confidence", 0.0)) * 100, 1)
            }
        return None
    if isinstance(preds, dict):
        items = sorted(preds.items(), key=lambda kv: kv[1], reverse=True)
        if items:
            c, conf = items[0]
            return {"label": str(c), "confidence": round(float(conf) * 100, 1)}
    return None

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    data = await image.read()
    resp = requests.post(
        CLASSIFY_URL,
        params={"api_key": ROBOFLOW_API_KEY},
        files={"file": ("image.jpg", data, image.content_type or "image/jpeg")},
        timeout=60,
    ).json()
    LAST["vision"] = resp
    LAST["vision_updated"] = datetime.utcnow().isoformat()
    return JSONResponse(resp)

# ---------------- Gas model ----------------
class GasReading(BaseModel):
    vrl: float | None = None
    adc: int | None = None
    adc_max: int | None = 1023
    vref: float | None = 5.0
    rl:   float | None = 10000.0
    rs:   float | None = None
    r0:   float | None = None

def _ppm_from_ratio(ratio: float, a: float, b: float) -> float:
    if ratio is None or ratio <= 0:
        return 0.0
    return max(0.0, a * (ratio ** b))

@app.post("/gas")
def gas(g: GasReading):
    VREF = float(g.vref or 5.0)
    RL   = float(g.rl or 10000.0)
    adc_val = g.adc

    if g.vrl is None and adc_val is not None:
        adc_max = int(g.adc_max or 1023)
        g.vrl = (float(adc_val) / float(adc_max)) * VREF

    if g.vrl is None and g.rs is None:
        return {"error": "Send at least one of: vrl, adc, or rs."}

    rs = float(g.rs) if g.rs is not None else ((VREF - g.vrl) * RL) / max(0.001, g.vrl)
    r0 = float(g.r0) if g.r0 is not None else rs
    ratio = rs / max(1e-6, r0)

    data = {
        "adc": adc_val,
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

# ---------------- Cron + History ----------------
@app.post("/cron/snapshot")
def cron_snapshot():
    if not LAST.get("gas") or not LAST["gas"].get("ppm"):
        return {"ok": False, "error": "No gas reading to snapshot yet."}
    save_reading(LAST["gas"]["ppm"])
    return {"ok": True, "saved": LAST["gas"]["ppm"]}

@app.get("/history")
def history():
    return {"history": load_history_last_days(days=2)}

# ---------------- Summary ----------------
def _summarize(last: dict) -> dict:
    pred = extract_top_class(last["vision"])
    gas = last["gas"] or {}
    ppm = gas.get("ppm", {})

    co2, nh3, benz, alco = ppm.get("co2"), ppm.get("nh3"), ppm.get("benzene"), ppm.get("alcohol")
    co2_hi = (co2 is not None) and (co2 >= 2000)
    nh3_hi = (nh3 is not None) and (nh3 >= 15)
    voc_hi = (benz or 0) >= 5 or (alco or 0) >= 10
    model_rotten = bool(pred and "rotten" in str(pred.get("label", "")).lower())
    spoiled = model_rotten or co2_hi or nh3_hi or voc_hi

    return {
        "vision": pred,
        "gas_ppm": {"co2": co2, "nh3": nh3, "benzene": benz, "alcohol": alco},
        "gas_raw": {
            "adc": gas.get("adc"),
            "vrl": gas.get("vrl"),
            "rs": gas.get("rs"),
            "r0": gas.get("r0"),
            "ratio": gas.get("ratio"),
        },
        "gas_flags": {"co2_high": co2_hi, "nh3_high": nh3_hi, "voc_high": voc_hi},
        "decision": "SPOILED" if spoiled else "FRESH",
    }

@app.get("/summary")
def summary():
    return _summarize(LAST)

# ---------------- UI ----------------
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

  <style>
    :root{
      --green:#22c55e; --green-dark:#16a34a;
      --blue:#0ea5e9; --amber:#f59e0b;
      --red:#ef4444; --slate:#1f2937;
      --glass-bg: rgba(255,255,255,.86);
      --glass-border: rgba(0,0,0,.06);
      --shadow: 0 10px 30px rgba(0,0,0,.18);
    }

    body{
      margin:0; font-family:Inter,system-ui,Arial,Helvetica,sans-serif; color:var(--slate);
      background:url('https://i.pinimg.com/originals/3d/91/51/3d9151870044e69f2d93a9d0311275dd.gif') center/cover no-repeat fixed; min-height:100vh
    }
    body::before{content:""; position:fixed; inset:0; background:linear-gradient(180deg,rgba(0,0,0,.32),rgba(0,0,0,.22)); pointer-events:none; z-index:-1}

    header{
      background:linear-gradient(90deg, rgba(34,197,94,.97), rgba(22,163,74,.97));
      padding:18px 16px; text-align:center; color:#fff; box-shadow:var(--shadow);
      position:sticky; top:0; z-index:10; backdrop-filter: blur(4px);
    }
    header h1{ margin:0; font-size:clamp(1.3rem,2.8vw,1.9rem); letter-spacing:.3px }
    header p{ margin:6px 0 0 0; opacity:.95 }

    .container{ width:92%; max-width:1100px; margin:24px auto; display:grid; gap:18px }

    .card{
      background:var(--glass-bg); border-radius:16px; padding:18px; box-shadow:var(--shadow);
      border:1px solid var(--glass-border); backdrop-filter: blur(6px);
      transition: transform .15s ease, box-shadow .2s ease;
    }
    .card:hover{ transform:translateY(-1px); box-shadow: 0 12px 34px rgba(0,0,0,.22); }
    .card h2{ margin:0 0 12px; color:var(--green-dark); font-size:1.15rem; border-left:4px solid var(--green-dark); padding-left:10px }

    button{
      background:var(--green); color:#fff; padding:10px 14px; border:none; border-radius:10px;
      cursor:pointer; font-weight:800; letter-spacing:.2px; box-shadow: 0 5px 18px rgba(34,197,94,.35);
      display:inline-flex; align-items:center; gap:8px; transition: background .15s ease, transform .1s ease;
    }
    button:hover{ background:var(--green-dark) }
    button:active{ transform:translateY(1px) }
    button.secondary{ background:#e7fff1; color:#0b3d2e; border:1px solid #b9f3d2; box-shadow:none; }
    button.gray{ background:#f3f4f6; color:#111827; border:1px solid #e5e7eb; box-shadow:none; }

    input[type=file], input[type=number]{
      padding:10px 12px; border:1px solid #d1d5db; border-radius:10px; background:#fff; font-weight:600;
      outline:none; transition:border-color .15s ease, box-shadow .15s ease;
    }
    input[type=file]:focus, input[type=number]:focus{ border-color: var(--green); box-shadow: 0 0 0 3px rgba(34,197,94,.18); }

    .row{ display:flex; gap:10px; flex-wrap:wrap; align-items:center }
    .pill{ padding:6px 10px; border-radius:999px; font-weight:700; font-size:.9rem; border:1px solid #d1d5db; background:#fff }
    .ok{ background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0 }
    .bad{ background:#fef2f2; color:#991b1b; border:1px solid #fecaca }
    .warn{ background:#fffbeb; color:#92400e; border:1px solid #fde68a }

    .big{ font-size:22px; font-weight:900; margin-top:10px }

    img,video,canvas{ max-width:100%; border-radius:12px; margin-top:10px }

    pre{
      white-space:pre-wrap; background:#0b1220; color:#e5e7eb; border-radius:12px; padding:12px; max-height:320px; overflow:auto;
      border:1px solid rgba(255,255,255,.05);
    }

    footer{ text-align:center; padding:16px; background:rgba(238,238,238,.92); color:#111827; margin-top:10px; font-size:.9rem; border-top:1px solid rgba(0,0,0,.06) }

    .chart-wrap{
      position:relative; height:280px; width:100%; overflow:hidden; border-radius:12px; background:#ffffffe6; border:1px solid rgba(0,0,0,.06)
    }
    .chart-empty{
      position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
      color:#6b7280; font-size:.95rem; pointer-events:none; font-weight:700;
    }

    .toast{
      position:fixed; right:14px; bottom:14px; background:rgba(17,24,39,.92); color:#fff; padding:10px 14px;
      border-radius:10px; font-weight:700; box-shadow: var(--shadow);
      transform: translateY(10px); opacity:0; transition: all .2s ease; z-index:30;
    }
    .toast.show{ transform: translateY(0); opacity:1; }

    .status{ font-size:.9rem; opacity:.85; display:inline-flex; align-items:center; gap:8px; margin-left:8px; }
    .dot{ width:8px; height:8px; border-radius:50%; background:var(--green) }
    .muted{ opacity:.7 }
  </style>
</head>
<body>
  <header>
    <h1>🍎 Fruit Freshness & Gas Detector</h1>
    <p>Upload, predict, and view gas-based decision</p>
  </header>

  <div class="container">
    <!-- Vision -->
    <div class="card">
      <h2>1) Upload or Capture Fruit Image <span id="visionStatus" class="status muted"><span class="dot"></span> idle</span></h2>
      <div class="row">
        <input id="file" type="file" accept="image/*" />
        <button onclick="predictFile()">🔮 Predict</button>
        <button class="secondary" onclick="startCam()">📷 Use Webcam</button>
        <button class="gray" onclick="snap()">📸 Snapshot</button>
        <button class="gray" onclick="stopCam()">⏹ Stop Camera</button>
        <button class="gray" onclick="clearVision()">🧹 Clear Image</button>
      </div>
      <video id="video" autoplay playsinline width="320" height="240" style="display:none;background:#000"></video>
      <canvas id="canvas" width="320" height="240" style="display:none"></canvas>
      <img id="preview" alt="preview" />
      <div id="visionTop" class="big"></div>
      <span id="visionBadge" class="pill" style="display:none"></span>
    </div>

    <!-- Gas -->
    <div class="card">
      <h2>2) Gas Sensor Reading <span id="gasStatus" class="status muted"><span class="dot" style="background:#0ea5e9"></span> idle</span></h2>
      <div class="row" style="margin-bottom:8px">
        ADC <input id="adc" type="number" value="1800" />
        Vref <input id="vref" type="number" value="5.0" step="0.1" />
        RL(Ω) <input id="rl" type="number" value="10000" />
        R0(Ω) <input id="r0" type="number" value="10000" />
        <button onclick="sendGas()">📤 Send</button>
        <button class="gray" onclick="preset('fresh')">🍏 Fresh Preset</button>
        <button class="gray" onclick="preset('spoiled')">🍌 Spoiled Preset</button>
        <button class="gray" onclick="resetGas()">🔁 Reset</button>
        <button class="gray" onclick="saveSnap()">💾 Save Snapshot</button>
      </div>
      <div id="gasBadges" style="margin-top:6px"></div>
    </div>

    <!-- Decision -->
    <div class="card">
      <h2>3) Final Decision</h2>
      <div id="decision" class="big"></div>
      <div class="row" style="margin:10px 0">
        <button class="secondary" onclick="refresh()">🔄 Refresh Summary</button>
        <button class="gray" onclick="clearAll()">🧹 Clear All</button>
      </div>
      <pre id="raw"></pre>
    </div>

    <!-- Chart -->
    <div class="card">
      <h2>4) Gas Chart (Last 2 Days)</h2>
      <div class="chart-wrap">
        <canvas id="gasChart"></canvas>
        <div id="chartEmpty" class="chart-empty">No readings yet — send gas data or save a snapshot.</div>
      </div>
    </div>
  </div>

  <footer>© 2025 Fruit Detector • FastAPI + Roboflow + MQ-135</footer>
  <div id="toast" class="toast">Saved ✔</div>

<script>
"use strict";
const badge = (t, c) => `<span class="pill ${c}">${t}</span>`;
function toast(msg){ const el=document.getElementById('toast'); el.textContent=msg||'Done'; el.classList.add('show'); setTimeout(()=>el.classList.remove('show'),1500); }

// Vision + webcam functions (unchanged) ...

// Gas form functions (unchanged) ...

// ---------- Summary (with ADC badge) ----------
async function refresh(){
  try {
    const r = await fetch('/summary', {cache:"no-store"});
    if(!r.ok) throw new Error(r.statusText);
    const s = await r.json();

    if (s.vision && s.vision.label){
      visionBadge.style.display='inline-block';
      const lbl = String(s.vision.label);
      const conf = Number(s.vision.confidence ?? 0).toFixed(1);
      const bad = /(^|_|\\b)rotten/i.test(lbl);
      visionBadge.className = 'pill ' + (bad ? 'bad' : 'ok');
      visionBadge.textContent = `${lbl} • ${conf}%`;
      visionTop.textContent = lbl.replaceAll('_',' ').toUpperCase();
    } else {
      visionBadge.style.display='none'; visionTop.textContent='';
    }

    const g = s.gas_ppm || {}, gf = s.gas_flags || {};
    gasBadges.innerHTML = [
      badge(`ADC ${s.gas_raw?.adc ?? '—'}`, 'pill'),
      badge(`CO₂ ${g.co2??'—'} ppm`, gf.co2_high ? 'bad' : 'ok'),
      badge(`NH₃ ${g.nh3??'—'} ppm`, gf.nh3_high ? 'bad' : 'ok'),
      badge(`VOC ${g.alcohol??'—'} eq`, gf.voc_high ? 'warn' : 'ok')
    ].join(' ');

    decision.className = 'big ' + (s.decision === 'SPOILED' ? 'bad' : 'ok');
    decision.textContent = s.decision || '';
    raw.textContent = JSON.stringify(s, null, 2);

  } catch(e){
    console.error("Refresh error", e);
    decision.className='big warn';
    decision.textContent='⚠ Backend offline';
  }
}
refresh(); setInterval(refresh, 2000);

// Chart.js load + update (unchanged except formatting) ...
</script>
</body>
</html>
    """
