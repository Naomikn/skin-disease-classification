from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import inference

app = FastAPI(title="Skin Disease Classifier")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
def health():
    return {
        "status":  "healthy",
        "model":   "ConvNeXt-Small + Swin-Small (AFF gate)",
        "classes": inference.NUM_CLASSES,
        "device":  str(inference.DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in [
        "image/jpeg", "image/jpg", "image/png", "image/webp"
    ]:
        return JSONResponse(
            status_code=400,
            content={"error": "Please upload a JPEG or PNG image."}
        )
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "Empty file uploaded."}
        )
    result = inference.predict_image(image_bytes)
    return JSONResponse(content=result)


@app.post("/flag")
async def flag():
    print("⚠️  Prediction flagged by clinician for review.")
    return {"status": "flagged"}
