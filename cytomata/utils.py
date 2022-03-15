import os
import imghdr
from natsort import natsorted


custom_palette = ['#648FFF', '#DC267F', '#785EF0', '#FE6100', '#FFB000']


custom_styles = {
    'image.cmap': 'turbo',
    'figure.figsize': (16, 8),
    'text.color': '#212121',
    'axes.titleweight': 'bold',
    'axes.titlesize': 46,
    'axes.titlepad': 18,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 42,
    'axes.labelpad': 10,
    'axes.labelcolor': '#212121',
    'axes.labelweight': 600,
    'axes.linewidth': 3,
    'axes.edgecolor': '#212121',
    'xtick.labelsize': 36,
    'ytick.labelsize': 36,
    'legend.fontsize': 36,
    'lines.linewidth': 5
}


def list_img_files(dir):
    return [os.path.join(dir, fn) for fn in natsorted(os.listdir(dir), key=lambda y: y.lower())
        if imghdr.what(os.path.join(dir, fn)) in ['tiff', 'jpeg', 'png', 'gif']]


def setup_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def clear_screen():
    os.system('cls' if os.name=='nt' else 'clear')
