from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from pathlib import Path
import cv2
from PIL import Image
import io
import numpy as np
from scipy import ndimage

app = FastAPI(
    title="Product Video Generator API",
    description="Tách nền ảnh sản phẩm"
)

# CORS - Cho phép request từ mọi nơi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạo folder
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Endpoint chính"""
    return {
        "message": "🎬 Product Video Generator API",
        "status": "running ✅",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok"}

@app.get("/info")
async def info():
    """Thông tin API"""
    return {
        "api": "Product Video Generator",
        "version": "1.0.0",
        "free": True,
        "features": [
            "Remove background from images",
            "Simple background segmentation",
            "Download processed images"
        ]
    }

@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    """
    API để tách nền ảnh
    """
    try:
        # Lưu file upload
        filename = file.filename
        input_path = f"{UPLOAD_DIR}/{filename}"
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Đọc ảnh bằng OpenCV
        img_cv = cv2.imread(input_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_array = img_rgb.astype(np.float32) / 255.0
        
        # Tính độ sáng
        lightness = np.mean(img_array, axis=2)
        
        # Tạo mask
        mask = lightness < 0.85
        
        # Làm mịn mask
        mask = ndimage.binary_closing(mask, iterations=2)
        mask = ndimage.binary_opening(mask, iterations=1)
        
        # Tạo RGBA
        result_array = np.array(img_rgb)
        result_array = np.dstack([result_array, np.ones(result_array.shape[:2]) * 255])
        result_array[~mask] = (255, 255, 255, 0)
        
        # Convert to PIL Image
        result = Image.fromarray(result_array.astype('uint8'), 'RGBA')
        
        # Lưu result
        output_filename = f"removed_{Path(filename).stem}.png"
        output_path = f"{OUTPUT_DIR}/{output_filename}"
        result.save(output_path, "PNG")
        
        return {
            "status": "success",
            "message": "✅ Tách nền thành công!",
            "filename": output_filename,
            "download_url": f"/download/{output_filename}"
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error: {str(e)}"
            }
        )

@app.post("/add-background")
async def add_background(
    file: UploadFile = File(...),
    bg_color: str = "white"
):
    """
    API để thêm background
    """
    try:
        filename = file.filename
        input_path = f"{UPLOAD_DIR}/{filename}"
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Đọc ảnh
        img_cv = cv2.imread(input_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        img = img.convert("RGBA")
        
        # Tạo background
        if bg_color == "white":
            bg = Image.new('RGB', img.size, (255, 255, 255))
        elif bg_color == "black":
            bg = Image.new('RGB', img.size, (0, 0, 0))
        else:
            bg = Image.new('RGB', img.size, (200, 200, 200))
        
        # Ghép ảnh
        bg.paste(img, (0, 0), img)
        
        # Lưu result
        output_filename = f"bg_{bg_color}_{Path(filename).stem}.png"
        output_path = f"{OUTPUT_DIR}/{output_filename}"
        bg.save(output_path, "PNG")
        
        return {
            "status": "success",
            "message": f"✅ Thêm background {bg_color} thành công!",
            "filename": output_filename,
            "download_url": f"/download/{output_filename}"
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/download/{filename}")
async def download(filename: str):
    """Download file đã xử lý"""
    file_path = f"{OUTPUT_DIR}/{filename}"
    
    if not os.path.exists(file_path):
        return JSONResponse(
            status_code=404,
            content={"error": "File not found"}
        )
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

# ==================== ERROR HANDLERS ====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)