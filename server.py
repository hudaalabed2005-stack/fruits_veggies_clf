from fastapi.responses import HTMLResponse
import os, requests
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

#  Roboflow config
ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]  # set in Render dashboard
PROJECT = "fresh-rotten-xvon4-9tefx"
VERSION = "1"
DETECT_URL = f"https://detect.roboflow.com/{PROJECT}/{VERSION}"

# App
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store last results
LAST = {
    "vision": None,
    "vision_updated": None,
    "gas": None,
    "gas_updated": None,
}

# ---------- IMAGE CLASSIFIER ----------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    data = await image.read()
    r = requests.post(
        DETECT_URL,
        params={"api_key": ROBOFLOW_API_KEY},
        files={"file": ("image.jpg", data, image.content_type or "image/jpeg")},
        timeout=60,
    )
    resp = r.json()
    LAST["vision"] = resp
    LAST["vision_updated"] = datetime.utcnow().isoformat()
    return JSONResponse(resp)

#GAS MODEL
class GasReading(BaseModel):
    vrl: float | None = None     # measured volts at RL
    adc: int | None = None       # raw ADC 0–4095
    vref: float | None = 3.3
    rl: float | None = 10000.0
    rs: float | None = None
    r0: float | None = None

def ppm(ratio: float, a: float, b: float):
    return max(0.0, a * (ratio ** b)) if ratio > 0 else 0.0

@app.post("/gas")
def gas(g: GasReading):
    VREF = g.vref or 3.3
    RL = g.rl or 10000.0
    if g.vrl is None and g.adc is not None:
        g.vrl = (g.adc / 4095.0) * VREF
    if g.vrl is None and g.rs is None:
        return {"error": "send vrl or adc or rs"}

    rs = g.rs if g.rs is not None else ((VREF - g.vrl) * RL) / max(0.001, g.vrl)
    r0 = g.r0 or rs
    ratio = rs / max(1e-6, r0)

    data = {
        "vrl": round(g.vrl, 3),
        "rs": round(rs, 1),
        "r0": round(r0, 1),
        "ratio": round(ratio, 3),
        "ppm": {
            "co2": round(ppm(ratio, 116.6021, -2.7690), 1),
            "nh3": round(ppm(ratio, 102.6940, -2.4880), 1),
            "benzene": round(ppm(ratio, 76.63, -2.1680), 1),
            "alcohol": round(ppm(ratio, 77.255, -3.18), 1),
        },
    }
    LAST["gas"] = data
    LAST["gas_updated"] = datetime.utcnow().isoformat()
    return {"ok": True, "data": data}

# ---------- RAW STATUS (both models) ----------
@app.get("/status")
def status():
    return {
        "vision": LAST["vision"],
        "vision_updated": LAST["vision_updated"],
        "gas": LAST["gas"],
        "gas_updated": LAST["gas_updated"],
    }

# SUMMARY (clean output + decision) 
def summarize(last):
    # pick top class from Roboflow result
    pred = None
    if last["vision"] and last["vision"].get("predictions"):
        preds = last["vision"]["predictions"]
        preds = sorted(preds, key=lambda p: p.get("confidence", 0), reverse=True)
        if preds:
            pred = {
                "label": preds[0].get("class", "?"),
                "confidence": round(float(preds[0].get("confidence", 0.0)) * 100, 1),
            }

    gas = last["gas"]["ppm"] if last["gas"] else {}
    co2 = gas.get("co2")
    nh3 = gas.get("nh3")
    benz = gas.get("benzene")
    alco = gas.get("alcohol")

    # simple thresholds (tune later)
    co2_hi = co2 is not None and co2 >= 800
    nh3_hi = nh3 is not None and nh3 >= 5
    voc_hi = (benz or 0) >= 0.4 or (alco or 0) >= 0.4

    # final decision: either model says rotten_* OR any gas high
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
    return summarize(LAST)

# Simple UI (friendlier badges + decision)
@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Fruit & Gas Demo</title>
  <style>
    body{font-family:system-ui,Arial,sans-serif;max-width:900px;margin:24px auto;padding:0 12px}
    .card{border:1px solid #e3e3e3;border-radius:14px;padding:16px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}
    button{padding:.6rem 1rem;border-radius:10px;border:1px solid #888;background:#fff;cursor:pointer}
    .pill{display:inline-block;padding:6px 10px;border-radius:999px;font-weight:600}
    .ok{background:#e8fff0;color:#147a2e;border:1px solid #b9e3c6}
    .warn{background:#fff5e6;color:#a05a00;border:1px solid #ffd9a6}
    .bad{background:#ffecec;color:#a40000;border:1px solid #ffb3b3}
    .big{font-size:28px;font-weight:800;margin:8px 0}
    .row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
    img{max-width:100%;border-radius:10px;margin-top:10px}
    pre{background:#fafafa;border:1px dashed #ddd;padding:10px;border-radius:10px;overflow:auto}
    input[type=number]{width:110px}
  </style>
</head>
<body>
  <h1>🍎 Fresh/Rotten + 🛡️ MQ-135</h1>

  <div class="card">
    <h2>1) Image prediction</h2>
    <div class="row">
      <input id="file" type="file" accept="image/*"/>
      <button onclick="predict()">Predict</button>
      <span id="visionBadge" class="pill ok" style="display:none"></span>
    </div>
    <div id="visionTop" class="big"></div>
    <img id="preview"/>
  </div>

  <div class="card">
    <h2>2) Gas reading (manual test)</h2>
    <div class="row">
      ADC <input id="adc" type="number" value="1800"/>
      Vref <input id="vref" type="number" value="3.3" step="0.1"/>
      RL(Ω) <input id="rl" type="number" value="10000"/>
      R0(Ω) <input id="r0" type="number" value="10000"/>
      <button onclick="sendGas()">Send</button>
    </div>
    <div id="gasBadges" style="margin-top:8px"></div>
  </div>

  <div class="card">
    <h2>3) Final decision</h2>
    <div id="decision" class="big"></div>
    <button onclick="refresh()">Refresh summary</button>
    <details style="margin-top:8px"><summary>Show raw</summary>
      <pre id="raw"></pre>
    </details>
  </div>

<script>
const badge = (txt, cls) => `<span class="pill ${cls}">${txt}</span>`;

async function predict(){
  const f = document.getElementById('file').files[0];
  if(!f){ alert('Choose an image'); return; }
  document.getElementById('preview').src = URL.createObjectURL(f);

  const fd = new FormData();
  fd.append('image', f, f.name);
  await fetch('/predict',{method:'POST', body: fd});
  await refresh(); // show updated summary
}

async function sendGas(){
  const body = {
    adc:  parseInt(document.getElementById('adc').value||'0'),
    vref: parseFloat(document.getElementById('vref').value||'3.3'),
    rl:   parseInt(document.getElementById('rl').value||'10000'),
    r0:   parseInt(document.getElementById('r0').value||'10000'),
  };
  await fetch('/gas',{method:'POST',
     headers:{'Content-Type':'application/json'},
     body: JSON.stringify(body)});
  await refresh();
}

async function refresh(){
  const res = await fetch('/summary');
  const s = await res.json();

  // Vision
  const vb = document.getElementById('visionBadge');
  const vt = document.getElementById('visionTop');
  if(s.vision){
    vb.style.display = 'inline-block';
    vb.className = 'pill ' + (String(s.vision.label).startsWith('rotten') ? 'bad' : 'ok');
    vb.textContent = `${s.vision.label} • ${s.vision.confidence}%`;
    vt.textContent = String(s.vision.label).replace('_',' ').toUpperCase();
  } else { vb.style.display = 'none'; vt.textContent=''; }

  // Gas
  const g = s.gas_ppm || {};
  const gf = s.gas_flags || {};
  const el = document.getElementById('gasBadges');
  el.innerHTML = [
    badge(`CO₂ ${g.co2 ?? '—'} ppm`,  gf.co2_high ? 'bad':'ok'),
    badge(`NH₃ ${g.nh3 ?? '—'} ppm`, gf.nh3_high ? 'bad':'ok'),
    badge(`VOC ${g.alcohol ?? '—'} eq`, (gf.voc_high ? 'warn':'ok'))
  ].join(' ');

  // Decision
  const d = document.getElementById('decision');
  d.className = 'big ' + (s.decision==='SPOILED' ? 'bad' : 'ok');
  d.textContent = s.decision;

  document.getElementById('raw').textContent = JSON.stringify(s, null, 2);
}
refresh();
</script>
</body>
</html>
    """
