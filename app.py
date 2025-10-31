import os
os.environ["USE_FLASH_ATTENTION"] = "0"  # ← ADD THIS AT THE VERY TOP

from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoModelForCausalLM
import torch
from PIL import Image
import io

app = FastAPI()

# Load model at startup
model = None
processor = None

@app.on_event("startup")
async def load_model():
    global model, processor
    print("🚀 Loading StarVector model...")
    try:
        # Force disable flash attention in multiple ways
        model = AutoModelForCausalLM.from_pretrained(
            "starvector/starvector-1b-im2svg",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            token=os.environ.get("HUGGING_FACE_HUB_TOKEN", None),
            use_flash_attention_2=False,  # Explicitly disable
            attn_implementation="eager"   # ← CRITICAL: Force eager attention
        )
        processor = model.model.processor
        model.eval()
        print("✅ StarVector model loaded!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        # Don't crash - let health endpoint handle it

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    # Your existing prediction code...
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    image_tensor = processor(image, return_tensors="pt")['pixel_values']
    if image_tensor.shape[0] != 1:
        image_tensor = image_tensor.squeeze(0)
    
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    batch = {"image": image_tensor}
    raw_svg = model.generate_im2svg(batch, max_length=4000)[0]
    
    return {"svg": raw_svg}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
