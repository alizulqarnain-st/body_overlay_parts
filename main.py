# ===== Libraries =====
import uuid
import imageio
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List
import os
import tempfile
import cv2
import shutil
import subprocess

from rembg import remove
from pydantic import BaseModel
from moviepy.editor import ImageSequenceClip

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ===== FastAPI App Setup =====
app = FastAPI(
    title="Body Overlay Parts",
    description="Overlay parts of garments or items on the transparent body using the given links of videos or images",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ===== Pydantic Schema =====
class VideoURLs(BaseModel):
    urls: List[str]

# ===== Merge Videos API Endpoint =====
@app.post("/merge-videos", response_class=FileResponse)
async def merge_videos(video_payload: VideoURLs):
    urls = video_payload.urls

    if len(urls) < 2:
        return {"error": "At least 2 videos required (body + overlays)"}

    body_index = next((i for i, url in enumerate(urls) if "bodyUploadVideo" in url), None)
    if body_index is None:
        return {"error": "Could not detect body video. Make sure one URL contains 'bodyUploadVideo'"}

    readers = [imageio.get_reader(url, format='ffmpeg') for url in urls]
    body_reader = readers[body_index]
    fps = body_reader.get_meta_data()['fps']
    frames = []

    for frame_set in zip(*readers):
        base_frame = frame_set[body_index][:, :, :3].astype(np.float32)

        for i, frame in enumerate(frame_set):
            if i == body_index:
                continue
            overlay_rgb = frame[:, :, :3].astype(np.float32)
            brightness = np.mean(overlay_rgb, axis=2)
            alpha = (brightness < 245).astype(np.float32)
            alpha = np.expand_dims(alpha, axis=2)
            base_frame = (1 - alpha) * base_frame + alpha * overlay_rgb

        frames.append(base_frame.astype(np.uint8))

    # Save merged video to temp directory
    temp_dir = tempfile.gettempdir()
    merged_path = os.path.join(temp_dir, f"merged_{uuid.uuid4().hex}.mp4")
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(merged_path, codec='libx264', audio=False, verbose=False, logger=None)

    # ===== Background Removal using rembg =====
    frame_temp_dir = os.path.join(temp_dir, "frames_" + uuid.uuid4().hex)
    output_frame_dir = os.path.join(temp_dir, "output_frames_" + uuid.uuid4().hex)
    os.makedirs(frame_temp_dir, exist_ok=True)
    os.makedirs(output_frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(merged_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frame_temp_dir, f"frame_{i:05d}.png")
        cv2.imwrite(frame_path, frame)
    cap.release()

    # Remove background using rembg with alpha
    frame_files = sorted(os.listdir(frame_temp_dir))
    for frame_file in frame_files:
        input_path = os.path.join(frame_temp_dir, frame_file)
        output_path = os.path.join(output_frame_dir, frame_file)

        with Image.open(input_path) as img:
            output_img = remove(img, alpha_matting=True, alpha_matting_foreground_threshold=240)
            output_img.save(output_path)

    # Encode video to webm with alpha channel
    output_webm = os.path.join(temp_dir, f"background_removed_{uuid.uuid4().hex}.webm")
    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(output_frame_dir, 'frame_%05d.png'),
        '-c:v', 'libvpx-vp9',
        '-pix_fmt', 'yuva420p',
        '-auto-alt-ref', '0',
        output_webm
    ], check=True)

    # Clean up temp frames
    shutil.rmtree(frame_temp_dir)
    shutil.rmtree(output_frame_dir)

    return FileResponse(output_webm, media_type="video/webm", filename="background_removed.webm")

class ImageURLs(BaseModel):
    image_urls : List[str]

# API EndPoint to merge the Images
@app.post("/merge-images")
async def merge_images(image_payload: ImageURLs):  # Asynchronous function to handle Image overlay using the Urls
    image_urls = image_payload.image_urls

    if len(image_urls) < 2:
        return {"error": "At least two images required (body + overlay)"}

    body_index = next((i for i, url in enumerate(image_urls) if 'bodyUploadImage' in url), None)

    if body_index is None:
        return {"error": "Couldn't detect Body Image. Make sure one URL contains 'bodyUploadImage'"}

    # Download and open all images
    images = []
    
    for url in image_urls:
    
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGBA")  # Ensure alpha channel
        images.append(img)

    base_img = images[body_index].convert("RGBA")

    for i, overlay in enumerate(images):
    
        if i == body_index:
            continue
    
        overlay_resized = overlay.resize(base_img.size)
        base_img = Image.alpha_composite(base_img, overlay_resized)

    output_filename = f"merged_image_output_{uuid.uuid4().hex}.png"
    base_img.save(output_filename, format="PNG")

    return FileResponse(output_filename, media_type="image/png", filename="merged_image_output.png")




# Just to test - remove after testing
# merge_images(urls)
# print(f"calling the function just to test{merge_images(urls)}")







## worked(removed final videos's background) but have left some white spaces near body
# # ===== Necessities - Libraries =====
# import uuid
# import imageio
# import requests
# import numpy as np
# from PIL import Image
# from io import BytesIO
# from typing import List
# import os
# import tempfile
# import cv2

# from pydantic import BaseModel
# from moviepy.editor import ImageSequenceClip

# from fastapi import FastAPI
# from fastapi.responses import FileResponse
# from fastapi.middleware.cors import CORSMiddleware

# # ===== FastAPI App Setup =====
# app = FastAPI(
#     title="Body Overlay Parts",
#     description="Overlay parts of garments or items on the transparent body using the given links of videos or images",
#     version="2.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

# # ===== Pydantic Schema =====
# class VideoURLs(BaseModel):
#     urls: List[str]

# # ===== Merge Videos API Endpoint =====
# @app.post("/merge-videos", response_class=FileResponse)
# async def merge_videos(video_payload: VideoURLs):
#     urls = video_payload.urls

#     if len(urls) < 2:
#         return {"error": "At least 2 videos required (body + overlays)"}

#     # Detect body video by keyword
#     body_index = next((i for i, url in enumerate(urls) if "bodyUploadVideo" in url), None)
#     if body_index is None:
#         return {"error": "Could not detect body video. Make sure one URL contains 'bodyUploadVideo'"}

#     # Read all videos
#     readers = [imageio.get_reader(url, format='ffmpeg') for url in urls]
#     body_reader = readers[body_index]
#     fps = body_reader.get_meta_data()['fps']
#     frames = []

#     # Overlay frames
#     for frame_set in zip(*readers):
#         base_frame = frame_set[body_index][:, :, :3].astype(np.float32)
#         for i, frame in enumerate(frame_set):
#             if i == body_index:
#                 continue
#             overlay_rgb = frame[:, :, :3].astype(np.float32)
#             brightness = np.mean(overlay_rgb, axis=2)
#             alpha = (brightness < 245).astype(np.float32)
#             alpha = np.expand_dims(alpha, axis=2)
#             base_frame = (1 - alpha) * base_frame + alpha * overlay_rgb
#         frames.append(base_frame.astype(np.uint8))

#     # Save merged video to temp dir
#     temp_dir = tempfile.gettempdir()
#     merged_path = os.path.join(temp_dir, f"merged_{uuid.uuid4().hex}.mp4")
#     clip = ImageSequenceClip(frames, fps=fps)
#     clip.write_videofile(merged_path, codec='libx264', audio=False, verbose=False, logger=None)

#     # ===== APPLY MOG2 BACKGROUND REMOVAL =====
#     import mediapipe as mp

#     cap = cv2.VideoCapture(merged_path)

#     mp_selfie_segmentation = mp.solutions.selfie_segmentation
#     with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:

#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)

#         output_path = os.path.join(tempfile.gettempdir(), f"final_bg_removed_{uuid.uuid4().hex}.mp4")
#         out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Convert to RGB
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = selfie_segmentation.process(frame_rgb)

#             # Create mask from segmentation
#             mask = results.segmentation_mask
#             condition = mask > 0.6  # Adjust threshold for better edge quality

#             # Keep only the foreground
#             bg_removed = np.zeros_like(frame)
#             bg_removed = np.where(condition[..., None], frame, 0)

#             out.write(bg_removed)

#         cap.release()
#         out.release()


#     return FileResponse(output_path, media_type="video/mp4", filename="background_removed.mp4")



# previous main code (first ever version)
# # Necessities - Libraries
# import uuid
# import imageio
# import requests
# import numpy as np
# from PIL import Image
# from io import BytesIO
# from typing import List

# from pydantic import BaseModel
# from moviepy.editor import ImageSequenceClip

# from fastapi import FastAPI
# from fastapi.responses import FileResponse
# from fastapi.middleware.cors import CORSMiddleware

# # Building the app using Fast API
# app = FastAPI(

#     title="Body Overlay Parts", 
#     description="Overlay parts of garmets or items on the transparent body using the given links of videos or images", 
#     version="2.0"

# )

# # CORs allowing
# app.add_middleware(

#     CORSMiddleware,
#     allow_origins = ["*"],
#     allow_credentials = True,
#     allow_methods = ["*"],
#     allow_headers = ["*"]

# )


# # Input urls either of videos or of images
# class VideoURLs(BaseModel):
#     urls: List[str]


# # Merge Videos API End-Point
# @app.post("/merge-videos", response_class=FileResponse)
# async def merge_videos(video_payload: VideoURLs):  # Asynchronous Function for merging videos using Urls
#     urls = video_payload.urls

#     # At least 2 urls required one should be body, if not will through error
#     if len(urls) < 2:
#         return {"error": "At least 2 videos required (body + overlays)"}

#     # Auto-detect body video (e.g., contains 'bodyUploadVideo')
#     body_index = next((i for i, url in enumerate(urls) if "bodyUploadVideo" in url), None)

#     if body_index is None:
#         return {"error": "Could not detect body video. Make sure one URL contains 'bodyUploadVideo'"}

#     # Read all videos
#     readers = [imageio.get_reader(url, format='ffmpeg') for url in urls]
#     body_reader = readers[body_index]
#     fps = body_reader.get_meta_data()['fps']
#     frames = []

#     for frame_set in zip(*readers):

#         base_frame = frame_set[body_index][:, :, :3].astype(np.float32)

#         for i, frame in enumerate(frame_set):

#             if i == body_index:
#                 continue  # Skip body

#             overlay_rgb = frame[:, :, :3].astype(np.float32)
#             brightness = np.mean(overlay_rgb, axis=2)
#             alpha = (brightness < 245).astype(np.float32)
#             alpha = np.expand_dims(alpha, axis=2)
#             base_frame = (1 - alpha) * base_frame + alpha * overlay_rgb

#         frames.append(base_frame.astype(np.uint8))

#     output_filename = f"merged_video_output_{uuid.uuid4().hex}.mp4"
#     clip = ImageSequenceClip(frames, fps=fps)
#     clip.write_videofile(output_filename, codec='libx264', audio=False, verbose=False, logger=None)

#     return FileResponse(output_filename, media_type="video/mp4", filename="merged_video_output.mp4")


# class ImageURLs(BaseModel):
#     image_urls : List[str]

# # API EndPoint to merge the Images
# @app.post("/merge-images")
# async def merge_images(image_payload: ImageURLs):  # Asynchronous function to handle Image overlay using the Urls
#     image_urls = image_payload.image_urls

#     if len(image_urls) < 2:
#         return {"error": "At least two images required (body + overlay)"}

#     body_index = next((i for i, url in enumerate(image_urls) if 'bodyUploadImage' in url), None)

#     if body_index is None:
#         return {"error": "Couldn't detect Body Image. Make sure one URL contains 'bodyUploadImage'"}

#     # Download and open all images
#     images = []
    
#     for url in image_urls:
    
#         response = requests.get(url)
#         img = Image.open(BytesIO(response.content)).convert("RGBA")  # Ensure alpha channel
#         images.append(img)

#     base_img = images[body_index].convert("RGBA")

#     for i, overlay in enumerate(images):
    
#         if i == body_index:
#             continue
    
#         overlay_resized = overlay.resize(base_img.size)
#         base_img = Image.alpha_composite(base_img, overlay_resized)

#     output_filename = f"merged_image_output_{uuid.uuid4().hex}.png"
#     base_img.save(output_filename, format="PNG")

#     return FileResponse(output_filename, media_type="image/png", filename="merged_image_output.png")
