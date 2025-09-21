from fastapi.responses import HTMLResponse
import os, requests
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]  # set in Render dashboard
PROJECT = "fresh-rotten-xvon4-9tefx"
VERSION = "1"
DETECT_URL = f"https://detect.roboflow.com/{PROJECT}/{VERSION}"

app = FastAPI()
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])

# Store last results
LAST = {"vision": None, "vision_updated": None,
        "gas": None, "gas_updated": None}

# IMAGE CLASSIFIER ENDPOINT
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    data = await image.read()
    r = requests.post(DETECT_URL,
                      params={"api_key": ROBOFLOW_API_KEY},
                      files={"file": ("image.jpg", data, image.content_type or "image/jpeg")},
                      timeout=60)
    resp = r.json()
    LAST["vision"] = resp
    LAST["vision_updated"] = datetime.utcnow().isoformat()
    return JSONResponse(resp)

# GAS MODEL ENDPOINT
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
        g.vrl = (g.adc/4095.0)*VREF
    if g.vrl is None and g.rs is None:
        return {"error": "send vrl or adc or rs"}
    rs = g.rs if g.rs is not None else ((VREF - g.vrl) * RL) / max(0.001, g.vrl)
    r0 = g.r0 or rs
    ratio = rs/max(1e-6, r0)

    data = {
        "vrl": round(g.vrl,3),
        "rs": round(rs,1),
        "r0": round(r0,1),
        "ratio": round(ratio,3),
        "ppm": {
            "co2": round(ppm(ratio,116.6021,-2.7690),1),
            "nh3": round(ppm(ratio,102.6940,-2.4880),1),
            "benzene": round(ppm(ratio,76.63,-2.1680),1),
            "alcohol": round(ppm(ratio,77.255,-3.18),1),
        }
    }
    LAST["gas"] = data
    LAST["gas_updated"] = datetime.utcnow().isoformat()
    return {"ok": True, "data": data}

# STATUS (both models)
@app.get("/status")
def status():
    return {"vision": LAST["vision"], "vision_updated": LAST["vision_updated"],
            "gas": LAST["gas"], "gas_updated": LAST["gas_updated"]}
@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Fruit & Gas Demo</title>
  <style>
    body{font-family:system-ui,Arial,sans-serif;max-width:920px;margin:2rem auto;padding:0 1rem}
    .card{border:1px solid #ddd;border-radius:12px;padding:16px;margin:12px 0}
    button{padding:.6rem 1rem;border-radius:10px;border:1px solid #888;cursor:pointer}
    #preview{max-width:100%;margin-top:8px}
    textarea{width:100%;height:180px}
    input[type=number]{width:100px}
  </style>
</head>
<body>
  <h1>🍎 Fresh/Rotten + 🛡️ MQ-135 (Cloud Demo)</h1>

  <div class="card">
    <h2>1) Image prediction</h2>
    <input id="file" type="file" accept="image/*" />
    <button onclick="predict()">Predict</button><br/>
    <img id="preview" />
    <h3>Result</h3>
    <pre id="out"></pre>
  </div>

  <div class="card">
    <h2>2) Gas reading (manual test)</h2>
    ADC: <input id="adc" type="number" value="1800"/> 
    Vref: <input id="vref" type="number" value="3.3" step="0.1"/>
    RL(Ω): <input id="rl" type="number" value="10000"/>
    R0(Ω): <input id="r0" type="number" value="10000"/>
    <button onclick="sendGas()">Send</button>
    <h3>Gas response</h3>
    <pre id="gasout"></pre>
  </div>

  <div class="card">
    <h2>3) Last combined status</h2>
    <button onclick="status()">Refresh status</button>
    <pre id="statusout"></pre>
  </div>

<script>
async function predict(){
  const f = document.getElementById('file').files[0];
  if(!f){ alert('Choose an image'); return; }
  document.getElementById('preview').src = URL.createObjectURL(f);

  const fd = new FormData();
  fd.append('image', f, f.name);
  const res = await fetch('/predict',{method:'POST', body: fd});
  const j = await res.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);
}

async function sendGas(){
  const body = {
    adc: parseInt(document.getElementById('adc').value || '0'),
    vref: parseFloat(document.getElementById('vref').value || '3.3'),
    rl: parseInt(document.getElementById('rl').value || '10000'),
    r0: parseInt(document.getElementById('r0').value || '10000')
  };
  const res = await fetch('/gas',{method:'POST',
     headers:{'Content-Type':'application/json'},
     body: JSON.stringify(body)});
  const j = await res.json();
  document.getElementById('gasout').textContent = JSON.stringify(j, null, 2);
}

async function status(){
  const res = await fetch('/status');
  const j = await res.json();
  document.getElementById('statusout').textContent = JSON.stringify(j, null, 2);
}
</script>
</body>
</html>
    """
