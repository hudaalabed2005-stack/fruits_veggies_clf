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
    raise RuntimeError("Missing ROBOFLOW_API_KEY environment variable.")

# ------------------ FastAPI app + CORS ------------------
app = FastAPI(title="Fruit & Gas Cloud API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    vrl: float | None = None
    adc: int | None = None
    adc_max: int | None = None
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
    VREF = g.vref or 3.3
    RL   = g.rl or 10000.0
    if g.vrl is None and g.adc is not None:
        adc_max = g.adc_max if g.adc_max is not None else (4095 if g.adc > 1023 else 1023)
        g.vrl = (g.adc / float(adc_max)) * VREF
    if g.vrl is None and g.rs is None:
        return {"error": "Send at least one of: vrl, adc, or rs."}
    rs = g.rs if g.rs is not None else ((VREF - g.vrl) * RL) / max(0.001, g.vrl)
    r0 = g.r0 or rs
    ratio = rs / max(1e-6, r0)
    data = {
        "vrl": round(g.vrl, 3) if g.vrl is not None else None,
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
    return {"ok": True, "data": data}

# ------------------ Status + Summary ------------------
@app.get("/status")
def status():
    return {
        "vision": LAST["vision"],
        "vision_updated": LAST["vision_updated"],
        "gas": LAST["gas"],
        "gas_updated": LAST["gas_updated"],
    }

def _summarize(last: dict) -> dict:
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
    co2, nh3, benz, alco = gas.get("co2"), gas.get("nh3"), gas.get("benzene"), gas.get("alcohol")
    co2_hi = (co2 is not None) and (co2 >= 2000)
    nh3_hi = (nh3 is not None) and (nh3 >= 15)
    voc_hi = (benz or 0) >= 5 or (alco or 0) >= 10
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
    return _summarize(LAST)

# ------------------ Welcome Page ------------------
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
      margin:0; font-family: Inter, Arial, sans-serif; color:#fff;
      min-height:100vh;
      background:url('https://i.pinimg.com/originals/30/ab/43/30ab43926be6852d3b03572459ab847d.gif') center center / cover no-repeat fixed;
      display:flex; align-items:center; justify-content:center;
    }
    body::before{
      content:""; position:fixed; inset:0;
      background:linear-gradient(to bottom right, rgba(0,0,0,.35), rgba(0,0,0,.55));
      pointer-events:none;
    }
    .wrap{ position:relative; text-align:center; padding:48px 40px;
      max-width:900px; width:92%; background:rgba(0,0,0,0.30);
      border:1px solid rgba(255,255,255,0.15); border-radius:20px;
      box-shadow:0 20px 60px rgba(0,0,0,.45); backdrop-filter: blur(6px);
    }
    h1{ font-size: clamp(2rem, 4vw, 3rem); margin:0 0 12px; }
    p{ font-size:1.05rem; opacity:.95; margin:0 auto 22px; max-width:760px; line-height:1.6 }
    .cta{ display:inline-block; margin-top:8px; padding:14px 26px; font-weight:800;
      color:#0b3d2e; background:linear-gradient(135deg,#7CFFCB,#4ADE80);
      border:none; border-radius:12px; text-decoration:none;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="badge">🍎 Fruit Freshness & Gas Detector</div>
    <h1>Smarter Food, Fresher Choices</h1>
    <p>Upload an image or use your webcam/phone camera, then send gas data to get a real-time decision.</p>
    <a class="cta" href="/app">Start Detecting Freshness</a>
  </div>
</body>
</html>
    """

# ------------------ App Page ------------------
@app.get("/app", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Fruit Freshness Detector</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
 body{ margin:0; font-family:Arial,sans-serif; background:#f3f4f6 }
 header{ background:#16a34a; padding:18px; text-align:center; color:#fff }
 .container{ width:92%; max-width:1100px; margin:24px auto }
 .card{ background:#fff; border-radius:12px; padding:18px; margin-bottom:18px; box-shadow:0 2px 6px rgba(0,0,0,0.1) }
 .row{ display:flex; gap:10px; flex-wrap:wrap; align-items:center }
 button{ background:#22c55e; color:#fff; padding:10px 14px; border:none; border-radius:8px; cursor:pointer; font-weight:700 }
 button.gray{ background:#f3f4f6; color:#111; border:1px solid #ddd }
</style>
</head>
<body>
<header>
  <h1>🍎 Fruit Freshness & Gas Detector</h1>
</header>
<div class="container">

  <div class="card">
    <h2>1) Upload or Capture Fruit Image</h2>
    <div class="row">
      <input id="file" type="file" accept="image/*" capture="environment" />
      <button onclick="predictFile()">Predict</button>
      <button onclick="startCam('environment')">Use Camera</button>
      <button class="gray" onclick="switchCamera()">Switch Camera</button>
      <button class="gray" onclick="snap()">Snapshot</button>
      <button class="gray" onclick="stopCam()">Stop Camera</button>
      <button class="gray" onclick="clearVision()">Clear Image</button>
    </div>
    <video id="video" autoplay playsinline muted width="320" height="240" style="display:none;background:#000"></video>
    <canvas id="canvas" width="320" height="240" style="display:none"></canvas>
    <img id="preview" alt="preview" />
    <div id="visionTop"></div>
    <span id="visionBadge" style="display:none"></span>
  </div>

  <div class="card">
    <h2>2) Gas Sensor Reading</h2>
    <div class="row">
      ADC <input id="adc" type="number" value="1800" />
      Vref <input id="vref" type="number" value="3.3" step="0.1" />
      RL <input id="rl" type="number" value="10000" />
      R0 <input id="r0" type="number" value="10000" />
      <button onclick="sendGas()">Send</button>
    </div>
    <div id="gasBadges"></div>
  </div>

  <div class="card">
    <h2>3) Final Decision</h2>
    <div id="decision"></div>
    <pre id="raw"></pre>
  </div>

</div>

<script>
let stream=null, currentFacing="environment";
async function startCam(facing="environment"){
  try{
    currentFacing=facing;
    if(stream) stopCam();
    stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:facing}},audio:false});
    video.srcObject=stream; video.style.display='block';
  }catch(e){alert('Camera error:'+e);}
}
function stopCam(){ if(stream){stream.getTracks().forEach(t=>t.stop()); stream=null;} video.style.display='none'; }
async function switchCamera(){ await startCam(currentFacing==="environment"?"user":"environment"); }
function clearVision(){ preview.src=''; preview.style.display='none'; video.style.display='none'; canvas.style.display='none'; visionBadge.style.display='none'; visionTop.textContent=''; }

async function predictFile(){
 const f=file.files[0]; if(!f){alert('Choose an image'); return;}
 preview.src=URL.createObjectURL(f); preview.style.display='block';
 const fd=new FormData(); fd.append('image',f,f.name);
 const r=await fetch('/predict',{method:'POST',body:fd}); const j=await r.json();
 let top=null; if(j.predictions&&j.predictions.length){ top=j.predictions.sort((a,b)=>(b.confidence||0)-(a.confidence||0))[0]; }
 if(top){ visionBadge.style.display='inline-block'; visionBadge.textContent=top.class+" • "+(top.confidence*100).toFixed(1)+"%"; visionTop.textContent=top.class.toUpperCase(); }
 await refresh();
}

function snap(){
 if(!stream){alert('Start the camera first'); return;}
 const ctx=canvas.getContext('2d'); canvas.style.display='block'; ctx.drawImage(video,0,0,canvas.width,canvas.height);
 canvas.toBlob(async b=>{ const fd=new FormData(); fd.append('image',b,'snapshot.jpg'); const r=await fetch('/predict',{method:'POST',body:fd}); const j=await r.json();
 let top=null; if(j.predictions&&j.predictions.length){ top=j.predictions.sort((a,b)=>(b.confidence||0)-(a.confidence||0))[0]; }
 if(top){ visionBadge.style.display='inline-block'; visionBadge.textContent=top.class+" • "+(top.confidence*100).toFixed(1)+"%"; visionTop.textContent=top.class.toUpperCase(); }
 await refresh(); },'image/jpeg',0.92);
}

async function sendGas(){
 const body={adc:parseInt(adc.value||'0'), vref:parseFloat(vref.value||'3.3'), rl:parseInt(rl.value||'10000'), r0:parseInt(r0.value||'10000')};
 await fetch('/gas',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); await refresh();
}

async function refresh(){ const r=await fetch('/summary'); const s=await r.json(); raw.textContent=JSON.stringify(s,null,2); decision.textContent=s.decision||''; }
setInterval(refresh,2000);
</script>
</body>
</html>
    """

