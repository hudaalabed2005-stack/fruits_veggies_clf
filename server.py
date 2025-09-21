# --- server.py ---------------------------------------------------------------
# FastAPI backend for your fruit (Roboflow) + MQ-135 gas project.
# Now with a webcam capture flow in the UI (in addition to file upload).
# -----------------------------------------------------------------------------

from fastapi.responses import HTMLResponse
import os, requests
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------------------------------
# Roboflow model configuration
# ------------------------------
ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]  # set in Render dashboard
PROJECT = "fresh-rotten-xvon4-9tefx"
VERSION = "1"
DETECT_URL = f"https://detect.roboflow.com/{PROJECT}/{VERSION}"

# ------------------------------
# FastAPI app + CORS
# ------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store the last results we got (just a tiny in-memory cache)
LAST = {
    "vision": None,
    "vision_updated": None,
    "gas": None,
    "gas_updated": None,
}

# ------------------------------
# IMAGE CLASSIFIER (Roboflow)
# ------------------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Accepts a multipart image (file or webcam snapshot),
    sends it to Roboflow, returns their JSON.
    """
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

# ------------------------------
# GAS ENDPOINT
# ------------------------------
class GasReading(BaseModel):
    # You can send either vrl (volts at RL), or adc (+ metadata)
    vrl: float | None = None
    adc: int | None = None
    adc_max: int | None = None   # optional override (4095 for ESP32, 1023 for UNO)
    vref: float | None = 3.3
    rl:   float | None = 10000.0
    rs:   float | None = None
    r0:   float | None = None

def ppm(ratio: float, a: float, b: float):
    # Tiny power law helper used in classic MQ-135 curves
    return max(0.0, a * (ratio ** b)) if ratio and ratio > 0 else 0.0

@app.post("/gas")
def gas(g: GasReading):
    """
    Accepts gas info from UNO/ESP32 (or your manual form).
    Computes Rs, ratio, and rough ppm for a few gases.
    Stores latest in LAST.
    """
    VREF = g.vref or 3.3
    RL   = g.rl or 10000.0

    # If only ADC is provided, compute VRL from it.
    if g.vrl is None and g.adc is not None:
        # Smart default for adc_max: if >1023 assume 12-bit, else 10-bit
        adc_max = g.adc_max if g.adc_max is not None else (4095 if g.adc > 1023 else 1023)
        g.vrl = (g.adc / float(adc_max)) * VREF

    if g.vrl is None and g.rs is None:
        return {"error": "send vrl or adc or rs"}

    # Compute Rs (sensor resistance) from voltage divider
    rs = g.rs if g.rs is not None else ((VREF - g.vrl) * RL) / max(0.001, g.vrl)
    r0 = g.r0 or rs
    ratio = rs / max(1e-6, r0)

    data = {
        "vrl": round(g.vrl, 3) if g.vrl is not None else None,
        "rs": round(rs, 1),
        "r0": round(r0, 1),
        "ratio": round(ratio, 3),
        "ppm": {
            "co2":     round(ppm(ratio, 116.6021, -2.7690), 1),
            "nh3":     round(ppm(ratio, 102.6940, -2.4880), 1),
            "benzene": round(ppm(ratio, 76.63,   -2.1680), 1),
            "alcohol": round(ppm(ratio, 77.255,  -3.18),   1),
        },
    }
    LAST["gas"] = data
    LAST["gas_updated"] = datetime.utcnow().isoformat()
    return {"ok": True, "data": data}

# ------------------------------
# STATUS + SUMMARY
# ------------------------------
@app.get("/status")
def status():
    """Raw latest objects (useful for debugging)."""
    return {
        "vision": LAST["vision"],
        "vision_updated": LAST["vision_updated"],
        "gas": LAST["gas"],
        "gas_updated": LAST["gas_updated"],
    }

def summarize(last: dict):
    """
    Squish the raw Roboflow JSON into a simple 'top label + confidence',
    attach gas ppm + simple flags, and produce a human decision.
    """
    # pick top class
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
    co2 = gas.get("co2"); nh3 = gas.get("nh3")
    benz = gas.get("benzene"); alco = gas.get("alcohol")

    # thresholds
    co2_hi = (co2 is not None) and (co2 >= 2000)     # ~2k ppm
    nh3_hi = (nh3 is not None) and (nh3 >= 15)       # ~15 ppm
    voc_hi = (benz or 0) >= 5 or (alco or 0) >= 10   # arbitrary "high VOC" bar

    model_rotten = bool(pred and str(pred["label"]).startswith("rotten"))
    spoiled = model_rotten or co2_hi or nh3_hi or voc_hi

    return {
        "vision": pred,
        "gas_ppm": {"co2": co2, "nh3": nh3, "benzene": benz, "alcohol": alco},
        "gas_flags": {"co2_high": co2_hi, "nh3_high": nh3_hi, "voc_high": voc_hi},
        "decision": "SPOILED" if spoiled else "FRESH",
    }

@app.get("/summary")
def summary():
    return summarize(LAST)

@app.get("/", response_class=HTMLResponse)
def ui():
e.
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Fruit & Gas (Cloud Demo)</title>
  <style>
    body{font-family:system-ui,Arial,sans-serif;max-width:980px;margin:28px auto;padding:0 14px}
    .card{border:1px solid #e5e5e5;border-radius:14px;padding:16px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}
    .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    button{padding:.55rem .9rem;border-radius:10px;border:1px solid #999;background:#fff;cursor:pointer}
    .pill{display:inline-block;padding:6px 10px;border-radius:999px;font-weight:600}
    .ok{background:#e8fff0;color:#147a2e;border:1px solid #b9e3c6}
    .warn{background:#fff5e6;color:#a05a00;border:1px solid #ffd9a6}
    .bad{background:#ffecec;color:#a40000;border:1px solid #ffb3b3}
    .big{font-size:28px;font-weight:800;margin:8px 0}
    img,video,canvas{max-width:100%;border-radius:10px}
    pre{background:#fafafa;border:1px dashed #ddd;padding:10px;border-radius:10px;overflow:auto}
    input[type=number], input[type=text]{padding:.3rem .4rem;border-radius:8px;border:1px solid #bbb}
    .tabbar{display:flex;gap:10px;margin-top:8px}
    .tabbar button{border-radius:999px}
    .hide{display:none}
  </style>
</head>
<body>
  <h1>🍎 Fresh/Rotten + 🛡️ MQ-135</h1>

  <!-- Vision card: upload OR webcam -->
  <div class="card">
    <h2>1) Image prediction</h2>

    <div class="tabbar">
      <button id="tabUpload"  onclick="showUpload()"  class="pill ok">Upload</button>
      <button id="tabWebcam"  onclick="showWebcam()"  class="pill">Webcam</button>
    </div>

    <!-- Upload mode -->
    <div id="uploadBox">
      <div class="row" style="margin-top:8px">
        <input id="file" type="file" accept="image/*"/>
        <button onclick="predictFile()">Predict</button>
        <span id="visionBadge" class="pill ok" style="display:none"></span>
      </div>
      <div id="visionTop" class="big"></div>
      <img id="preview"/>
    </div>

    <!-- Webcam mode -->
    <div id="webcamBox" class="hide" style="margin-top:10px">
      <div class="row">
        <button onclick="startCam()">Open camera</button>
        <button onclick="snap()" id="snapBtn" disabled>Capture</button>
        <button onclick="predictSnap()" id="predictSnapBtn" disabled>Predict snapshot</button>
        <span id="visionBadge2" class="pill ok" style="display:none"></span>
      </div>
      <div class="row" style="margin-top:8px">
        <video id="video" autoplay playsinline width="480" height="360" style="background:#000"></video>
        <canvas id="canvas" class="hide" width="480" height="360"></canvas>
      </div>
      <div id="visionTop2" class="big"></div>
      <img id="snapPreview" />
    </div>
  </div>

  <!-- Gas card (manual poke for testing) -->
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

  <!-- Combined decision -->
  <div class="card">
    <h2>3) Final decision</h2>
    <div id="decision" class="big"></div>
    <div class="row">
      <button onclick="refresh()">Refresh summary</button>
    </div>
    <details style="margin-top:8px"><summary>Show raw</summary>
      <pre id="raw"></pre>
    </details>
  </div>

<script>
/* ------------------ little helper UI bits ------------------ */
const badge = (txt, cls) => `<span class="pill ${cls}">${txt}</span>`;
function showUpload(){
  document.getElementById('uploadBox').classList.remove('hide');
  document.getElementById('webcamBox').classList.add('hide');
  document.getElementById('tabUpload').classList.add('ok');
  document.getElementById('tabWebcam').classList.remove('ok');
}
function showWebcam(){
  document.getElementById('webcamBox').classList.remove('hide');
  document.getElementById('uploadBox').classList.add('hide');
  document.getElementById('tabWebcam').classList.add('ok');
  document.getElementById('tabUpload').classList.remove('ok');
}

/* ------------------ file upload flow ------------------ */
async function predictFile(){
  const f = document.getElementById('file').files[0];
  if(!f){ alert('Choose an image'); return; }
  document.getElementById('preview').src = URL.createObjectURL(f);

  const fd = new FormData();
  fd.append('image', f, f.name);
  const res = await fetch('/predict',{method:'POST', body: fd});
  const j = await res.json();

  // pretty badge + title
  const vb = document.getElementById('visionBadge');
  const vt = document.getElementById('visionTop');
  let top = null;
  if(j && j.predictions && j.predictions.length){
    top = j.predictions.sort((a,b)=> (b.confidence||0)-(a.confidence||0))[0];
  }
  if(top){
    vb.style.display='inline-block';
    const lbl = String(top.class||'?');
    vb.className = 'pill ' + (lbl.startsWith('rotten')?'bad':'ok');
    vb.textContent = `${lbl} • ${(top.confidence*100).toFixed(1)}%`;
    vt.textContent = lbl.replace('_',' ').toUpperCase();
  }
  await refresh();
}

/* ------------------ webcam flow ------------------ */
let stream = null, lastSnapBlob = null;

async function startCam(){
  try{
    stream = await navigator.mediaDevices.getUserMedia({video:true, audio:false});
    const video = document.getElementById('video');
    video.srcObject = stream;
    document.getElementById('snapBtn').disabled = false;
  }catch(e){
    alert('Could not open camera: ' + e);
  }
}

function snap(){
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  canvas.classList.remove('hide');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(b=>{
    lastSnapBlob = b;
    document.getElementById('snapPreview').src = URL.createObjectURL(b);
    document.getElementById('predictSnapBtn').disabled = false;
  }, 'image/jpeg', 0.92);
}

async function predictSnap(){
  if(!lastSnapBlob){ alert('Take a snapshot first'); return; }
  const fd = new FormData();
  fd.append('image', lastSnapBlob, 'snapshot.jpg');
  const res = await fetch('/predict',{method:'POST', body: fd});
  const j = await res.json();

  const vb = document.getElementById('visionBadge2');
  const vt = document.getElementById('visionTop2');
  let top = null;
  if(j && j.predictions && j.predictions.length){
    top = j.predictions.sort((a,b)=> (b.confidence||0)-(a.confidence||0))[0];
  }
  if(top){
    vb.style.display='inline-block';
    const lbl = String(top.class||'?');
    vb.className = 'pill ' + (lbl.startsWith('rotten')?'bad':'ok');
    vb.textContent = `${lbl} • ${(top.confidence*100).toFixed(1)}%`;
    vt.textContent = lbl.replace('_',' ').toUpperCase();
  }
  await refresh();
}

/* ------------------ gas test button (manual) ------------------ */
async function sendGas(){
  const body = {
    adc:  parseInt(document.getElementById('adc').value||'0'),
    vref: parseFloat(document.getElementById('vref').value||'3.3'),
    rl:   parseInt(document.getElementById('rl').value||'10000'),
    r0:   parseInt(document.getElementById('r0').value||'10000')
  };
  const res = await fetch('/gas',{method:'POST',
     headers:{'Content-Type':'application/json'},
     body: JSON.stringify(body)});
  await res.json();
  await refresh();
}

/* ------------------ summary refresh + pretty badges ------------------ */
async function refresh(){
  const res = await fetch('/summary');
  const s = await res.json();

  // Gas badges
  const g = s.gas_ppm || {};
  const gf = s.gas_flags || {};
  const el = document.getElementById('gasBadges');
  el.innerHTML = [
    badge(`CO₂ ${g.co2 ?? '—'} ppm`,  gf.co2_high ? 'bad':'ok'),
    badge(`NH₃ ${g.nh3 ?? '—'} ppm`, gf.nh3_high ? 'bad':'ok'),
    badge(`VOC ${g.alcohol ?? '—'} eq`, (gf.voc_high ? 'warn':'ok'))
  ].join(' ');

  // Final decision
  const d = document.getElementById('decision');
  d.className = 'big ' + (s.decision==='SPOILED' ? 'bad' : 'ok');
  d.textContent = s.decision;

  document.getElementById('raw').textContent = JSON.stringify(s, null, 2);
}

// show upload by default + auto-poll summary every 2s
showUpload();
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
    """
