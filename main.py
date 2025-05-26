from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import imageio
import numpy as np
from moviepy.editor import ImageSequenceClip
import uuid

app = FastAPI()

# ✅ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoURLs(BaseModel):
    urls: List[str]

@app.post("/merge-videos")
async def merge_videos(payload: VideoURLs):
    video_urls = payload.urls

    if len(video_urls) < 2:
        return {"error": "At least 2 videos required (body + overlays)"}

    # Auto-detect body video (e.g., contains 'bodyUploadVideo')
    body_index = next((i for i, url in enumerate(video_urls) if "bodyUploadVideo" in url), None)

    if body_index is None:
        return {"error": "Could not detect body video. Make sure one URL contains 'bodyUploadVideo'"}

    # Read all videos
    readers = [imageio.get_reader(url, format='ffmpeg') for url in video_urls]
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

    output_filename = f"merged_output_{uuid.uuid4().hex}.mp4"
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_filename, codec='libx264', audio=False, verbose=False, logger=None)

    return FileResponse(output_filename, media_type="video/mp4", filename="merged_output.mp4")
