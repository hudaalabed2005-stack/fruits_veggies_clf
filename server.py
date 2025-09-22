import os
from datetime import datetime

import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

#Roboflow
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
PROJECT = "fresh-rotten-xvon4-9tefx"
VERSION = "1"
DETECT_URL = f"https://detect.roboflow.com/{PROJECT}/{VERSION}"

if not ROBOFLOW_API_KEY:
    raise RuntimeError("Missing ROBOFLOW_API_KEY environment variable.")

# FastAPI app
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

# store history of gas readings
HISTORY = []

#predict
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Accept a multipart image (upload OR webcam snapshot),
    forward to Roboflow, and remember the last prediction.
    """
    data = await image.read()
    resp = requests.post(
        DETECT_URL,
        params={"api_key": ROBOFLOW_API_KEY},
        files={"file": ("image.jpg", data, image.content_type or "image/jpeg")},
        timeout=60,
    ).json()

    LAST["vision"] = resp
    LAST["vision_updated"] = datetime.utcnow().isoformat()
    return JSONResponse(resp)

# gas
class GasReading(BaseModel):
    # either vrl or adc
    vrl: float | None = None
    adc: int | None = None
    adc_max: int | None = None   # 4095 (ESP32) or 1023 (UNO) if sending "adc"
    vref: float | None = 3.3
    rl:   float | None = 10000.0
    rs:   float | None = None    # precomputed Rs (rarely used)
    r0:   float | None = None    # baseline in clean air (important for ppm)

def _ppm_from_ratio(ratio: float, a: float, b: float) -> float:
    """Classic MQ-135 power-law curve helper."""
    if ratio is None or ratio <= 0:
        return 0.0
    return max(0.0, a * (ratio ** b))

@app.post("/gas")
def gas(g: GasReading):
    """
    Accept gas info from ESP32/UNO (or the manual UI),
    compute Rs/ratio/ppm and remember the latest.
    """
    VREF = g.vref or 3.3
    RL   = g.rl or 10000.0

    # If only ADC is provided, compute VRL from it.
    if g.vrl is None and g.adc is not None:
        # Smart default for adc_max: assume 12-bit if value > 1023
        adc_max = g.adc_max if g.adc_max is not None else (4095 if g.adc > 1023 else 1023)
        g.vrl = (g.adc / float(adc_max)) * VREF

    if g.vrl is None and g.rs is None:
        return {"error": "Send at least one of: vrl, adc, or rs."}

    # Rs from divider if not provided
    rs = g.rs if g.rs is not None else ((VREF - g.vrl) * RL) / max(0.001, g.vrl)
    r0 = g.r0 or rs                      # if unknown, treat current as baseline
    ratio = rs / max(1e-6, r0)

    data = {
        "vrl": round(g.vrl, 3) if g.vrl is not None else None,
        "rs": round(rs, 1),
        "r0": round(r0, 1),
        "ratio": round(ratio, 3),
        "ppm": {
            # These "a, b" are ballpark for MQ-135; tune to your calibration.
            "co2":     round(_ppm_from_ratio(ratio, 116.6021, -2.7690), 1),
            "nh3":     round(_ppm_from_ratio(ratio, 102.6940, -2.4880), 1),
            "benzene": round(_ppm_from_ratio(ratio, 76.63,   -2.1680), 1),
            "alcohol": round(_ppm_from_ratio(ratio, 77.255,  -3.18),   1),
        },
    }

    LAST["gas"] = data
    LAST["gas_updated"] = datetime.utcnow().isoformat()

    # save to history
    HISTORY.append({"time": datetime.utcnow().isoformat(), "ppm": data["ppm"]})

    return {"ok": True, "data": data}

# Status
@app.get("/status")
def status():
    """Raw objects (handy for debugging)."""
    return {
        "vision": LAST["vision"],
        "vision_updated": LAST["vision_updated"],
        "gas": LAST["gas"],
        "gas_updated": LAST["gas_updated"],
    }

@app.get("/history")
def history():
    """Return last 2 days of gas readings"""
    cutoff = datetime.utcnow().timestamp() - 2*24*60*60
    last2days = [h for h in HISTORY if datetime.fromisoformat(h["time"]).timestamp() >= cutoff]
    return {"history": last2days}

def _summarize(last: dict) -> dict:
    """Compress raw Roboflow JSON + gas into one friendly decision."""
    # Top vision label + confidence
    pred = None
    if last["vision"] and last["vision"].get("predictions"):
        preds = sorted(
            last["vision"]["predictions"],
            key=lambda p: p.get("confidence", 0.0),
            reverse=True,
        )
        if preds:
            pred = {
                "label": preds[0].get("class", "?"),
                "confidence": round(float(preds[0].get("confidence", 0.0)) * 100, 1),
            }

    gas = last["gas"]["ppm"] if last["gas"] else {}
    co2  = gas.get("co2")
    nh3  = gas.get("nh3")
    benz = gas.get("benzene")
    alco = gas.get("alcohol")

    # Thresholds you will tune with real data
    co2_hi = (co2 is not None) and (co2 >= 2000)     # ~2k ppm
    nh3_hi = (nh3 is not None) and (nh3 >= 15)       # ~15 ppm
    voc_hi = (benz or 0) >= 5 or (alco or 0) >= 10   # crude VOC flag

    model_rotten = bool(pred and str(pred["label"]).startswith("rotten"))
    spoiled = model_rotten or co2_hi or nh3_hi or voc_hi

    return {
        "vision": pred,  # e.g. {"label":"fresh_apple","confidence":91.5}
        "gas_ppm": {"co2": co2, "nh3": nh3, "benzene": benz, "alcohol": alco},
        "gas_flags": {"co2_high": co2_hi, "nh3_high": nh3_hi, "voc_high": voc_hi},
        "decision": "SPOILED" if spoiled else "FRESH",
    }

@app.get("/summary")
def summary():
    return _summarize(LAST)

# 1) Welcome page at "/"
@app.get("/", response_class=HTMLResponse)
def welcome():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Welcome • Fruit Detector</title>
  <style>
    body{margin:0;font-family:Inter,Arial,Helvetica,sans-serif;color:#fff;min-height:100vh;
    background:url('https://i.pinimg.com/originals/30/ab/43/30ab43926be6852d3b03572459ab847d.gif')
    center center / cover no-repeat fixed;display:flex;align-items:center;justify-content:center;}
    body::before{content:"";position:fixed;inset:0;
    background:linear-gradient(to bottom right,rgba(0,0,0,.35),rgba(0,0,0,.55));pointer-events:none;}
    .wrap{position:relative;text-align:center;padding:48px 40px;max-width:900px;width:92%;
    background:rgba(0,0,0,0.30);border:1px solid rgba(255,255,255,0.15);border-radius:20px;
    box-shadow:0 20px 60px rgba(0,0,0,.45);backdrop-filter:blur(6px);}
    h1{font-size:clamp(2rem,4vw,3rem);margin:0 0 12px;letter-spacing:.5px;}
    p{font-size:1.05rem;opacity:.95;margin:0 auto 22px;max-width:760px;line-height:1.6}
    .cta{display:inline-block;margin-top:8px;padding:14px 26px;font-weight:800;letter-spacing:.2px;
    color:#0b3d2e;background:linear-gradient(135deg,#7CFFCB,#4ADE80);border:none;border-radius:12px;
    text-decoration:none;box-shadow:0 8px 24px rgba(16,185,129,.35);transition:transform .15s ease;}
    .cta:hover{transform:translateY(-2px);box-shadow:0 12px 28px rgba(16,185,129,.45);}
    .badge{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border-radius:999px;
    font-weight:700;color:#0b3d2e;background:rgba(255,255,255,.9);}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="badge"><span class="emoji">🍎</span><span>Fruit Freshness & Gas Detector</span></div>
    <h1>Smarter Food, Fresher Choices</h1>
    <p>Check fruit freshness with AI and estimate air quality using your sensor readings.</p>
    <a class="cta" href="/app">Start Detecting Freshness</a>
    <div class="meta">Or explore the API docs at <code>/docs</code>.</div>
  </div>
</body>
</html>
    """

# 2) Main app page at "/app"
@app.get("/app", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fruit Freshness Detector</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body{margin:0;font-family:Arial,Helvetica,sans-serif;color:#1f2937;
        background:url('https://i.pinimg.com/originals/3d/91/51/3d9151870044e69f2d93a9d0311275dd.gif') center center/cover no-repeat fixed;min-height:100vh;}
        body::before{content:"";position:fixed;inset:0;background:rgba(0,0,0,0.35);pointer-events:none;z-index:-1;}
        header{background:linear-gradient(90deg,rgba(34,197,94,0.95),rgba(22,163,74,0.95));padding:18px;text-align:center;color:#fff}
        .container{width:92%;max-width:1100px;margin:24px auto}
        .card{background:rgba(255,255,255,0.94);border-radius:12px;padding:18px;margin-bottom:18px;box-shadow:0 2px 6px rgba(0,0,0,0.1);}
        .card h2{margin:0 0 12px;color:#16a34a;font-size:1.2rem;border-left:4px solid #16a34a;padding-left:10px;}
        button{background:#22c55e;color:#fff;padding:10px 14px;border:none;border-radius:8px;cursor:pointer;font-weight:700}
        button:hover{background:#16a34a}
        input[type=file],input[type=number]{padding:8px;border:1px solid #d1d5db;border-radius:6px;background:#fff}
        .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
        .pill{padding:6px 10px;border-radius:999px;font-weight:600;font-size:0.9rem;border:1px solid #d1d5db;background:#fff}
        .ok{background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0}
        .bad{background:#fef2f2;color:#991b1b;border:1px solid #fecaca}
        .warn{background:#fffbeb;color:#92400e;border:1px solid #fde68a}
        .big{font-size:22px;font-weight:800;margin-top:10px}
        canvas{max-width:100%;margin-top:10px}
    </style>
</head>
<body>
<header>
    <h1>🍎 Fruit Freshness & Gas Detector</h1>
    <p>Simple, clear interface — upload, predict, and view gas-based decision</p>
</header>
<div class="container">

  <!-- Vision card -->
  <div class="card">
    <h2>1) Upload or Capture Fruit Image</h2>
    <div class="row">
      <input id="file" type="file" accept="image/*" />
      <button onclick="predictFile()">Predict</button>
      <button class="secondary" onclick="startCam()">Use Webcam</button>
      <button class="gray" onclick="snap()">Snapshot</button>
      <button class="gray" onclick="stopCam()">Stop Camera</button>
      <button class="gray" onclick="clearVision()">Clear Image</button>
    </div>
    <video id="video" autoplay playsinline width="320" height="240" style="display:none;background:#000"></video>
    <canvas id="canvas" width="320" height="240" style="display:none"></canvas>
    <img id="preview" alt="preview" />
    <div id="visionTop" class="big"></div>
    <span id="visionBadge" class="pill" style="display:none"></span>
  </div>

  <!-- Gas card -->
  <div class="card">
    <h2>2) Gas Sensor Reading</h2>
    <div class="row" style="margin-bottom:8px">
      ADC <input id="adc" type="number" value="1800" />
      Vref <input id="vref" type="number" value="3.3" step="0.1" />
      RL(Ω) <input id="rl" type="number" value="10000" />
      R0(Ω) <input id="r0" type="number" value="10000" />
      <button onclick="sendGas()">Send</button>
      <button class="gray" onclick="preset('fresh')">Fresh Preset</button>
      <button class="gray" onclick="preset('spoiled')">Spoiled Preset</button>
      <button class="gray" onclick="resetGas()">Reset</button>
    </div>
    <div id="gasBadges" style="margin-top:6px"></div>
  </div>

  <!-- Decision card -->
  <div class="card">
    <h2>3) Final Decision</h2>
    <div id="decision" class="big"></div>
    <div class="row" style="margin:10px 0">
      <button class="secondary" onclick="refresh()">Refresh Summary</button>
      <button class="gray" onclick="clearAll()">Clear All</button>
    </div>
    <pre id="raw"></pre>
  </div>

  <!-- Chart card -->
  <div class="card">
    <h2>4) Gas Chart (Last 2 Days)</h2>
    <canvas id="gasChart" height="120"></canvas>
  </div>

</div>
<footer>© 2025 Fruit Detector • FastAPI + Roboflow + MQ-135</footer>

<script>
const badge=(t,c)=>`<span class="pill ${c}">${t}</span>`;

function clearVision(){preview.src='';preview.style.display='none';video.style.display='none';canvas.style.display='none';visionBadge.style.display='none';visionTop.textContent='';}
function clearAll(){clearVision();gasBadges.innerHTML='';decision.className='big';decision.textContent='';raw.textContent='';}

async function predictFile(){const f=file.files[0];if(!f){alert('Choose an image');return;}
preview.src=URL.createObjectURL(f);preview.style.display='block';const fd=new FormData();fd.append('image',f,f.name);
const r=await fetch('/predict',{method:'POST',body:fd});const j=await r.json();let top=null;
if(j.predictions&&j.predictions.length){top=j.predictions.sort((a,b)=>(b.confidence||0)-(a.confidence||0))[0];}
if(top){visionBadge.style.display='inline-block';const lbl=String(top.class||'?');visionBadge.className='pill '+(lbl.startsWith('rotten')?'bad':'ok');visionBadge.textContent=`${lbl} • ${(top.confidence*100).toFixed(1)}%`;visionTop.textContent=lbl.replace('_',' ').toUpperCase();}
await refresh();}

let stream=null;async function startCam(){try{stream=await navigator.mediaDevices.getUserMedia({video:true
