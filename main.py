from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
from runModel import runModel
import numpy as np
import cv2
app = FastAPI()

# Enable CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, set your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your PyTorch model
model = torch.load("image_classifier.pt", map_location=torch.device("cpu"))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # 2. Convert bytes data to a NumPy array of type uint8
    nparr = np.frombuffer(contents, np.uint8)

    # 3. Decode image data to an OpenCV image (BGR format by default)
    cvImg = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    
    # Transform image as needed for your model
    prediction=runModel(image,cvImg)
    return {"result": prediction}


