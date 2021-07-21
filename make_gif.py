import imageio
import os
from natsort import os_sorted

image_folder = os.path.join('plots', 'elliptical slice sampling', 'gifs', 'images' )
img_list = os.listdir(image_folder)
img_list = os_sorted(img_list)

with imageio.get_writer('movie.gif', mode='I') as writer:
    for filename in img_list:
        print(filename)
        image = imageio.imread(os.path.join(image_folder, filename))
        writer.append_data(image)