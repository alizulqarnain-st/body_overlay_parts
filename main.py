# Necessities - Libraries
import uuid
import imageio
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List

from pydantic import BaseModel
from moviepy.editor import ImageSequenceClip

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Building the app using Fast API
app = FastAPI(

    title="Body Overlay Parts", 
    description="Overlay parts of garmets or items on the transparent body using the given links of videos or images", 
    version="2.0"

)

# CORs allowing
app.add_middleware(

    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]

)


# Input urls either of videos or of images
class VideoURLs(BaseModel):
    urls: List[str]


# Merge Videos API End-Point
@app.post("/merge-videos", response_class=FileResponse)
async def merge_videos(video_payload: VideoURLs):  # Asynchronous Function for merging videos using Urls
    urls = video_payload.urls

    # At least 2 urls required one should be body, if not will through error
    if len(urls) < 2:
        return {"error": "At least 2 videos required (body + overlays)"}

    # Auto-detect body video (e.g., contains 'bodyUploadVideo')
    body_index = next((i for i, url in enumerate(urls) if "bodyUploadVideo" in url), None)

    if body_index is None:
        return {"error": "Could not detect body video. Make sure one URL contains 'bodyUploadVideo'"}

    # Read all videos
    readers = [imageio.get_reader(url, format='ffmpeg') for url in urls]
    body_reader = readers[body_index]
    fps = body_reader.get_meta_data()['fps']
    frames = []

    for frame_set in zip(*readers):

        base_frame = frame_set[body_index][:, :, :3].astype(np.float32)

        for i, frame in enumerate(frame_set):

            if i == body_index:
                continue  # Skip body

            overlay_rgb = frame[:, :, :3].astype(np.float32)
            brightness = np.mean(overlay_rgb, axis=2)
            alpha = (brightness < 245).astype(np.float32)
            alpha = np.expand_dims(alpha, axis=2)
            base_frame = (1 - alpha) * base_frame + alpha * overlay_rgb

        frames.append(base_frame.astype(np.uint8))

    output_filename = f"merged_video_output_{uuid.uuid4().hex}.mp4"
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_filename, codec='libx264', audio=False, verbose=False, logger=None)

    return FileResponse(output_filename, media_type="video/mp4", filename="merged_video_output.mp4")


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
