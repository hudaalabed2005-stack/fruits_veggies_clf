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
