import imageio
import os

def convert(name):
    parts = name.split('_')
    converted = float(parts[0] + '.' + parts[1][0:-4])
    return converted

filenames = sorted(os.listdir('./gif'), key=convert)
images = [imageio.imread('./gif/'+filename) for filename in filenames]

imageio.mimwrite(os.path.join('slice_sampling.gif'), images, duration=.2)
