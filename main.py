from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pathlib import Path
import tempfile
import asyncio
from PIL import Image, ImageDraw, ImageFilter
import cv2
import os
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
import numpy as np
from moviepy.editor import ImageClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.VideoClip import VideoClip
import imageio

# Tạo app FastAPI
app = FastAPI(
    title="Product Video Generator API",
    description="Tách nền ảnh, thêm background, tạo video chuyển động",
    version="1.0.0"
)

# CORS - Cho phép request từ mọi nơi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạo folder lưu file
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== BACKGROUND REMOVAL ====================

def remove_background(image_path: str) -> Image.Image:
    """Tách nền từ ảnh bằng color segmentation"""
    try:
        from scipy import ndimage
        from skimage.color import rgb2hsv
        
        # Đọc ảnh
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Tính lightness
        lightness = np.mean(img_array, axis=2)
        
        # Tạo mask - pixel nào là object (không phải nền trắng/sáng)
        mask = lightness < 0.85
        
        # Làm mịn mask
        mask = ndimage.binary_closing(mask, iterations=2)
        mask = ndimage.binary_opening(mask, iterations=1)
        
        # Tạo output RGBA
        result_array = np.array(img.convert('RGBA'))
        result_array[~mask] = (255, 255, 255, 0)
        
        return Image.fromarray(result_array)
    except Exception as e:
        print(f"Error removing background: {e}")
        raise

# ==================== BACKGROUND OPERATIONS ====================

def create_solid_background(size: tuple, color: str = "white") -> Image.Image:
    """Tạo background solid color"""
    if color == "white":
        bg = Image.new('RGB', size, (255, 255, 255))
    elif color == "black":
        bg = Image.new('RGB', size, (0, 0, 0))
    elif color == "gradient":
        # Tạo gradient từ xanh dương đến tím
        width, height = size
        bg = Image.new('RGB', size)
        pixels = bg.load()
        for y in range(height):
            ratio = y / height
            r = int(102 * (1 - ratio) + 150 * ratio)
            g = int(126 * (1 - ratio) + 100 * ratio)
            b = int(234 * (1 - ratio) + 200 * ratio)
            for x in range(width):
                pixels[x, y] = (r, g, b)
        return bg
    else:
        bg = Image.new('RGB', size, (200, 200, 200))
    return bg

def composite_images(foreground_path: str, bg_color: str = "white") -> Image.Image:
    """Ghép foreground với background"""
    try:
        fg = Image.open(foreground_path).convert("RGBA")
        
        # Tạo background
        bg = create_solid_background(fg.size, bg_color)
        
        # Ghép ảnh
        bg.paste(fg, (0, 0), fg)
        
        return bg.convert("RGB")
    except Exception as e:
        print(f"Error compositing images: {e}")
        raise

def add_glow_effect(image_path: str, intensity: float = 0.3) -> Image.Image:
    """Thêm hiệu ứng glow"""
    try:
        img = Image.open(image_path)
        
        # Tạo blur
        blurred = img.filter(ImageFilter.GaussianBlur(radius=10))
        
        # Blend ảnh gốc với blur
        result = Image.blend(img, blurred, intensity)
        
        return result
    except Exception as e:
        print(f"Error adding glow: {e}")
        raise

def add_shadow_effect(image_path: str) -> Image.Image:
    """Thêm shadow dưới ảnh"""
    try:
        img = Image.open(image_path).convert("RGBA")
        
        # Tạo shadow
        shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(shadow)
        
        # Vẽ shadow
        width, height = img.size
        for i in range(height // 2, height):
            alpha = int(50 * (1 - (i - height // 2) / (height // 2)))
            draw.line([(0, i), (width, i)], fill=(0, 0, 0, alpha))
        
        # Composite
        result = Image.new('RGBA', img.size)
        result.paste(img, (0, 0), img)
        result.paste(shadow, (0, 0), shadow)
        
        return result.convert("RGB")
    except Exception as e:
        print(f"Error adding shadow: {e}")
        raise

# ==================== VIDEO GENERATION ====================

def create_rotating_video(
    image_path: str,
    duration: int = 5,
    fps: int = 30,
    output_path: str = "output.mp4"
) -> str:
    """Tạo video xoay 360 độ"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Tính số frame
        total_frames = duration * fps
        
        # Hàm tạo frame có rotation
        def make_frame(t):
            angle = (t / duration) * 360
            
            # Xoay ảnh
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            rotated = cv2.warpAffine(img_array, M, (w, h))
            
            return rotated
        
        # Tạo video
        frames = []
        for i in range(total_frames):
            t = i / fps
            frame = make_frame(t)
            frames.append(frame)
        
        # Lưu video
        writer = imageio.get_writer(output_path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        
        return output_path
    except Exception as e:
        print(f"Error creating rotating video: {e}")
        raise

def create_zoom_video(
    image_path: str,
    duration: int = 5,
    fps: int = 30,
    output_path: str = "output.mp4"
) -> str:
    """Tạo video zoom in/out"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        total_frames = duration * fps
        
        def make_frame(t):
            # Zoom từ 0.8 đến 1.2
            scale = 0.8 + (t / duration) * 0.4
            
            # Tính kích thước mới
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            # Resize ảnh
            resized = cv2.resize(img_array, (new_w, new_h))
            
            # Tạo canvas
            canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
            
            # Paste ở giữa
            y_offset = (h - new_h) // 2
            x_offset = (w - new_w) // 2
            
            if y_offset >= 0 and x_offset >= 0:
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        
        frames = []
        for i in range(total_frames):
            t = i / fps
            frame = make_frame(t)
            frames.append(frame)
        
        writer = imageio.get_writer(output_path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        
        return output_path
    except Exception as e:
        print(f"Error creating zoom video: {e}")
        raise

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Endpoint chính"""
    return {
        "message": "🎬 Product Video Generator API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "remove_background": "/remove-background",
            "add_background": "/add-background",
            "create_video": "/create-video",
            "full_process": "/full-process"
        }
    }

@app.post("/remove-background")
async def remove_bg_endpoint(file: UploadFile = File(...)):
    """
    API để tách nền ảnh
    
    Returns:
    - filename: Tên file ảnh đã tách nền
    - download_url: URL để download
    """
    try:
        # Lưu file upload
        input_path = f"{UPLOAD_DIR}/{file.filename}"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Tách nền
        output_image = remove_background(input_path)
        
        # Lưu result
        output_filename = f"removed_{Path(file.filename).stem}.png"
        output_path = f"{OUTPUT_DIR}/{output_filename}"
        output_image.save(output_path, "PNG")
        
        return {
            "status": "success",
            "message": "Tách nền thành công!",
            "filename": output_filename,
            "download_url": f"/download/{output_filename}"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/add-background")
async def add_bg_endpoint(
    file: UploadFile = File(...),
    bg_type: str = "white"
):
    """
    API để thêm background
    
    Parameters:
    - file: Ảnh sản phẩm (có nền trong suốt)
    - bg_type: white, black, gradient
    """
    try:
        # Lưu file upload
        input_path = f"{UPLOAD_DIR}/{file.filename}"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ghép background
        output_image = composite_images(input_path, bg_type)
        
        # Lưu result
        output_filename = f"bg_{bg_type}_{Path(file.filename).stem}.png"
        output_path = f"{OUTPUT_DIR}/{output_filename}"
        output_image.save(output_path, "PNG")
        
        return {
            "status": "success",
            "message": f"Thêm background {bg_type} thành công!",
            "filename": output_filename,
            "download_url": f"/download/{output_filename}"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/create-video")
async def create_video_endpoint(
    file: UploadFile = File(...),
    video_type: str = "rotate",
    duration: int = 5
):
    """
    API để tạo video
    
    Parameters:
    - file: Ảnh sản phẩm
    - video_type: rotate, zoom
    - duration: Thời lượng video (giây)
    """
    try:
        # Lưu file upload
        input_path = f"{UPLOAD_DIR}/{file.filename}"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Tạo video
        output_filename = f"video_{video_type}_{Path(file.filename).stem}.mp4"
        output_path = f"{OUTPUT_DIR}/{output_filename}"
        
        if video_type == "rotate":
            create_rotating_video(input_path, duration, output_path=output_path)
        elif video_type == "zoom":
            create_zoom_video(input_path, duration, output_path=output_path)
        
        return {
            "status": "success",
            "message": f"Tạo video {video_type} thành công!",
            "filename": output_filename,
            "download_url": f"/download/{output_filename}"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/full-process")
async def full_process(
    file: UploadFile = File(...),
    bg_type: str = "gradient",
    add_glow: bool = True,
    video_type: str = "rotate",
    video_duration: int = 5
):
    """
    API hoàn chỉnh: Tách nền → Thêm background → Thêm hiệu ứng → Tạo video
    
    Parameters:
    - file: Ảnh sản phẩm
    - bg_type: white, black, gradient
    - add_glow: Có thêm glow effect không
    - video_type: rotate, zoom
    - video_duration: Thời lượng video
    """
    try:
        filename = file.filename
        stem = Path(filename).stem
        
        # Bước 1: Tách nền
        input_path = f"{UPLOAD_DIR}/{filename}"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"1️⃣ Tách nền...")
        removed_bg = remove_background(input_path)
        removed_path = f"{UPLOAD_DIR}/removed_{stem}.png"
        removed_bg.save(removed_path, "PNG")
        
        # Bước 2: Thêm background
        print(f"2️⃣ Thêm background {bg_type}...")
        result_with_bg = composite_images(removed_path, bg_type)
        bg_path = f"{UPLOAD_DIR}/bg_{stem}.png"
        result_with_bg.save(bg_path, "PNG")
        
        # Bước 3: Thêm hiệu ứng
        if add_glow:
            print(f"3️⃣ Thêm glow effect...")
            result_with_fx = add_glow_effect(bg_path)
            fx_path = f"{UPLOAD_DIR}/fx_{stem}.png"
            result_with_fx.save(fx_path, "PNG")
        else:
            fx_path = bg_path
        
        # Bước 4: Tạo video
        print(f"4️⃣ Tạo video {video_type}...")
        output_filename = f"final_{video_type}_{stem}.mp4"
        output_path = f"{OUTPUT_DIR}/{output_filename}"
        
        if video_type == "rotate":
            create_rotating_video(fx_path, video_duration, output_path=output_path)
        elif video_type == "zoom":
            create_zoom_video(fx_path, video_duration, output_path=output_path)
        
        print(f"✅ Hoàn thành!")
        
        return {
            "status": "success",
            "message": "Xử lý hoàn toàn thành công!",
            "steps": [
                "✅ Tách nền",
                f"✅ Thêm background {bg_type}",
                "✅ Thêm hiệu ứng" if add_glow else "⏭️ Bỏ qua hiệu ứng",
                f"✅ Tạo video {video_type} ({video_duration}s)"
            ],
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
    """Download file"""
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

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "ok",
        "message": "API is running"
    }

@app.get("/info")
async def info():
    """Thông tin về API"""
    return {
        "api": "Product Video Generator",
        "version": "1.0.0",
        "features": [
            "Remove background from images",
            "Add custom backgrounds (white, black, gradient)",
            "Add visual effects (glow, shadow)",
            "Generate rotating videos",
            "Generate zoom videos",
            "Full automated pipeline"
        ],
        "free_tier": "Yes ✅",
        "max_file_size": "100MB"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)