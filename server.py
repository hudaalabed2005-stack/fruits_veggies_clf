# ------------------------------ server.py -----------------------------------
# Cloud backend for your Fruit (Roboflow) + MQ-135 project.
# - /predict : send an image (file or webcam snapshot) -> Roboflow model JSON
# - /gas     : send gas reading (vrl or adc) -> compute ppm + store "last"
# - /status  : raw latest vision + gas
# - /summary : human-friendly decision from both
# - /        : simple UI (Upload tab + Webcam tab + Gas test + Summary)
# ---------------------------------------------------------------------------

import os
from datetime import datetime

import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ------------------ Roboflow configuration ------------------
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
PROJECT = "fresh-rotten-xvon4-9tefx"
VERSION = "1"
DETECT_URL = f"https://detect.roboflow.com/{PROJECT}/{VERSION}"

if not ROBOFLOW_API_KEY:
    # Fail-fast with a clear message if the key isn't set in Render env vars
    raise RuntimeError("Missing ROBOFLOW_API_KEY environment variable.")

# ------------------ FastAPI app + CORS ------------------
app = FastAPI(title="Fruit & Gas Cloud API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # okay for demo; lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ tiny in-memory cache ------------------
LAST = {
    "vision": None,
    "vision_updated": None,
    "gas": None,
    "gas_updated": None,
}

# ------------------ Vision: /predict ------------------
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

# ------------------ Gas: /gas ------------------
class GasReading(BaseModel):
    # You may send either vrl (volts at RL) OR adc (+ metadata).
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
    return {"ok": True, "data": data}

# ------------------ Status + Summary ------------------
@app.get("/status")
def status():
    """Raw objects (handy for debugging)."""
    return {
        "vision": LAST["vision"],
        "vision_updated": LAST["vision_updated"],
        "gas": LAST["gas"],
        "gas_updated": LAST["gas_updated"],
    }

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

# ------------------ UI (Upload + Webcam + Gas test + Summary) ------------------
@app.get("/", response_class=HTMLResponse)
def ui():
    # Keep this as one triple-quoted string; indentation matters only for "return".
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

  <!-- 1) Vision -->
  <div class="card">
    <h2>1) Image prediction</h2>
    <div class="tabbar">
      <button id="tabUpload" onclick="showUpload()" class="pill ok">Upload</button>
      <button id="tabWebcam" onclick="showWebcam()" class="pill">Webcam</button>
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
        <button id="snapBtn" onclick="snap()" disabled>Capture</button>
        <button id="predictSnapBtn" onclick="predictSnap()" disabled>Predict snapshot</button>
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

  <!-- 2) Gas -->
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

  <!-- 3) Decision -->
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
/* ---------- tiny UI helpers ---------- */
const badge = (t,c)=>`<span class="pill ${c}">${t}</span>`;
function showUpload(){ uploadBox.classList.remove('hide'); webcamBox.classList.add('hide'); tabUpload.classList.add('ok'); tabWebcam.classList.remove('ok'); }
function showWebcam(){ webcamBox.classList.remove('hide'); uploadBox.classList.add('hide'); tabWebcam.classList.add('ok'); tabUpload.classList.remove('ok'); }

/* ---------- upload flow ---------- */
async function predictFile(){
  const f = file.files[0];
  if(!f){ alert('Choose an image'); return; }
  preview.src = URL.createObjectURL(f);
  const fd = new FormData(); fd.append('image', f, f.name);
  const r = await fetch('/predict',{method:'POST', body: fd}); const j = await r.json();

  let top=null;
  if(j && j.predictions && j.predictions.length){
    top = j.predictions.sort((a,b)=>(b.confidence||0)-(a.confidence||0))[0];
  }
  if(top){
    visionBadge.style.display='inline-block';
    const lbl = String(top.class||'?');
    visionBadge.className = 'pill ' + (lbl.startsWith('rotten')?'bad':'ok');
    visionBadge.textContent = `${lbl} • ${(top.confidence*100).toFixed(1)}%`;
    visionTop.textContent = lbl.replace('_',' ').toUpperCase();
  }
  await refresh();
}

/* ---------- webcam flow ---------- */
let stream=null,lastSnapBlob=null;

async function startCam(){
  try{
    stream = await navigator.mediaDevices.getUserMedia({video:true,audio:false});
    video.srcObject = stream;
    snapBtn.disabled = false;
  }catch(e){ alert('Could not open camera: ' + e); }
}
function snap(){
  const ctx = canvas.getContext('2d');
  canvas.classList.remove('hide');
  ctx.drawImage(video,0,0,canvas.width,canvas.height);
  canvas.toBlob(b=>{
    lastSnapBlob = b;
    snapPreview.src = URL.createObjectURL(b);
    predictSnapBtn.disabled = true; // enable after a tick to avoid double-click
    predictSnapBtn.disabled = false;
  }, 'image/jpeg', 0.92);
}
async function predictSnap(){
  if(!lastSnapBlob){ alert('Take a snapshot first'); return; }
  const fd = new FormData(); fd.append('image', lastSnapBlob, 'snapshot.jpg');
  const r = await fetch('/predict',{method:'POST', body: fd}); const j = await r.json();

  let top=null;
  if(j && j.predictions && j.predictions.length){
    top = j.predictions.sort((a,b)=>(b.confidence||0)-(a.confidence||0))[0];
  }
  if(top){
    visionBadge2.style.display='inline-block';
    const lbl = String(top.class||'?');
    visionBadge2.className = 'pill ' + (lbl.startsWith('rotten')?'bad':'ok');
    visionBadge2.textContent = `${lbl} • ${(top.confidence*100).toFixed(1)}%`;
    visionTop2.textContent = lbl.replace('_',' ').toUpperCase();
  }
  await refresh();
}

/* ---------- gas manual test ---------- */
async function sendGas(){
  const body = {
    adc:  parseInt(adc.value||'0'),
    vref: parseFloat(vref.value||'3.3'),
    rl:   parseInt(rl.value||'10000'),
    r0:   parseInt(r0.value||'10000')
  };
  await fetch('/gas',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  await refresh();
}

/* ---------- summary refresh ---------- */
async function refresh(){
  const r = await fetch('/summary'); const s = await r.json();
  const g = s.gas_ppm || {}, gf = s.gas_flags || {};

  gasBadges.innerHTML = [
    badge(`CO₂ ${g.co2 ?? '—'} ppm`,  gf.co2_high ? 'bad':'ok'),
    badge(`NH₃ ${g.nh3 ?? '—'} ppm`, gf.nh3_high ? 'bad':'ok'),
    badge(`VOC ${g.alcohol ?? '—'} eq`, gf.voc_high ? 'warn':'ok')
  ].join(' ');

  decision.className = 'big ' + (s.decision==='SPOILED' ? 'bad' : 'ok');
  decision.textContent = s.decision;
  raw.textContent = JSON.stringify(s, null, 2);
}

showUpload();        // default tab
refresh();           // initial summary
setInterval(refresh, 2000);  // auto-update every 2s
</script>
</body>
</html>
    """
# ---------------------------- end server.py ----------------------------------
