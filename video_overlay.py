# Standalone Function of Video Overlay (not connected with main)
import imageio
import numpy as np
from moviepy.editor import ImageSequenceClip
import uuid

def merge_videos_from_urls(video_urls):
    if len(video_urls) < 2:
        raise ValueError("At least two videos required (body + overlays)")

    # Auto-detect body video by URL keyword
    body_index = next((i for i, url in enumerate(video_urls) if "bodyUploadVideo" in url), None)
    if body_index is None:
        raise ValueError("Could not detect body video. Make sure one URL contains 'bodyUploadVideo'")

    # Read videos using imageio
    readers = [imageio.get_reader(url, format='ffmpeg') for url in video_urls]
    body_reader = readers[body_index]
    fps = body_reader.get_meta_data()['fps']
    frames = []

    for frame_set in zip(*readers):
        # Start with body frame as base
        base_frame = frame_set[body_index][:, :, :3].astype(np.float32)

        # Blend overlays
        for i, frame in enumerate(frame_set):
            if i == body_index:
                continue
            overlay_rgb = frame[:, :, :3].astype(np.float32)

            # Simple brightness-based alpha mask
            brightness = np.mean(overlay_rgb, axis=2)
            alpha = (brightness < 245).astype(np.float32)
            alpha = np.expand_dims(alpha, axis=2)

            base_frame = (1 - alpha) * base_frame + alpha * overlay_rgb

        frames.append(base_frame.astype(np.uint8))

    # Save output
    output_filename = f"merged_video_output_{uuid.uuid4().hex}.mp4"
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_filename, codec='libx264', audio=False, verbose=False, logger=None)

    print(f"✅ Merged video saved as: {output_filename}")
    return output_filename



# Just for testing - Remove it afterwards
# urls = [
#     "https://videomergerbackend.projectsaeedan.com/item_video/1740748036.webm",
#     "https://videomergerbackend.projectsaeedan.com/bodyUploadVideo/1739536037.webm"
# ]

