import ffmpeg
import numpy as np

def mask():
    from PIL import Image
    import numpy as np

    # Set the dimensions of the image
    width, height = 800, 600

    # Create a numpy array filled with dark pixels
    image_data = np.full((height, width, 3), 20, dtype=np.uint8)

    # Create a PIL Image object from the numpy array
    image = Image.fromarray(image_data)

    # Save the image to a file
    image.save('dark_image.png')
# Define the input video stream
input_stream = ffmpeg.input('720p.mp4')

# Define the region to be ignored
# overlay_file = ffmpeg.input('overlay.png')
overlay_file = ffmpeg.input('dark_image.png')

(
    ffmpeg
    .concat(
        input_stream.trim(start_frame=10, end_frame=20),
        input_stream.trim(start_frame=30, end_frame=40),
    )
    # .overlay(overlay_file.hflip())
    # .drawbox(50, 50, 120, 120, color='red', thickness=5)
    .output('out.mp4')
    .run()
)