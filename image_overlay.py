# # Standalone function of Image Overlay ( )
# import requests
# from io import BytesIO
# from PIL import Image
# import uuid

# def merge_images_from_urls(image_urls):
#     if len(image_urls) < 2:
#         raise ValueError("At least two images required (body + overlay)")

#     # Auto-detect body image
#     body_index = next((i for i, url in enumerate(image_urls) if 'bodyUploadImage' in url), None)

#     if body_index is None:
#         raise ValueError("Couldn't detect body image. Make sure one URL contains 'bodyUploadImage'")

#     # Load all images as RGBA (to support transparency)
#     images = []
#     for url in image_urls:
#         response = requests.get(url)
#         image = Image.open(BytesIO(response.content)).convert("RGBA")
#         images.append(image)

#     # Use body image as base
#     base_img = images[body_index].convert("RGBA")

#     # Overlay all other images
#     for i, overlay in enumerate(images):
#         if i == body_index:
#             continue
#         overlay_resized = overlay.resize(base_img.size)
#         base_img = Image.alpha_composite(base_img, overlay_resized)

#     # Save output
#     output_filename = f"merged_image_output_{uuid.uuid4().hex}.png"
#     base_img.save(output_filename)
#     print(f"✅ Merged image saved as: {output_filename}")
#     return output_filename



# Just for testing - remove 
# urls = ["https://admin.tabletopavatarforge.com/item_image/1747661280.png","https://admin.tabletopavatarforge.com/bodyUploadImage/1747661252.png"]
# merge_images_from_urls(urls)

import imageio
import numpy as np
from matplotlib import pyplot

image = input("Please enter the image url: ")
print(f"Just logging in for testing {image} ")

image_real = plt.imshow(image)
plt.title("Test Overlay Image")
plt.axis('off')
plt.imshow()
