from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoModelForCausalLM
import torch
from PIL import Image
import io
import os

app = FastAPI()

# Load model at startup
model = None
processor = None

@app.on_event("startup")
async def load_model():
    global model, processor
    print("🚀 Loading StarVector model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "starvector/starvector-1b-im2svg",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            token=os.environ.get("HUGGING_FACE_HUB_TOKEN", None),
            use_flash_attention_2=False  # ← FIX: Disable flash attention
        )
        processor = model.model.processor
        model.eval()
        print("✅ StarVector model loaded!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
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
