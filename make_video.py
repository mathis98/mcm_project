from natsort import os_sorted
import cv2
import os

image_folder = os.path.join('plots', 'elliptical slice sampling', 'gifs', 'images' )
image_folder = os.path.join('plots', 'slice sampling', 'gifs', 'images' )
video_name = 'video2.avi'

img_list = os.listdir(image_folder)
fps = 24

images = [img for img in os_sorted(img_list)]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, fps, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

