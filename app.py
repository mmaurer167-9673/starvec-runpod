from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from transformers import AutoModelForCausalLM
import torch
from PIL import Image
import io
import os

app = FastAPI()
security = HTTPBearer()

# Load model at startup
model = None
processor = None

# Your API key - set via environment variable
API_KEY = os.environ.get("API_KEY", "your-secret-key-here")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

@app.on_event("startup")
async def load_model():
    global model, processor
    print("🚀 Loading StarVector model...")
    model = AutoModelForCausalLM.from_pretrained(
        "starvector/starvector-1b-im2svg",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        token=os.environ.get("HUGGING_FACE_HUB_TOKEN", None)
    )
    processor = model.model.processor
    model.eval()
    print("✅ StarVector model loaded!")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    authorized: bool = Depends(verify_token)
):
    # Read image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Process and generate
    image_tensor = processor(image, return_tensors="pt")['pixel_values']
    if image_tensor.shape[0] != 1:
        image_tensor = image_tensor.squeeze(0)
    
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    batch = {"image": image_tensor}
    raw_svg = model.generate_im2svg(batch, max_length=4000)[0]
    
    return {"svg": raw_svg}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "StarVector API is running!"}
