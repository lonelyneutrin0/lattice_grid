import os
import cv2
# Path to the folder containing images
image_folder = 'frames'
output_video = 'output_video.mp4'
frame_rate = 60  # Frames per second

# Get list of all image files in the folder, sorted in order
images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
print(images)
# Read the first image to get the frame size
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Loop through images and write them to the video
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# Release the VideoWriter object
video.release()
print("Video created successfully!")