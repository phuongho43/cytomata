import os
import imghdr
from natsort import natsorted


custom_palette = ['#648FFF', '#785EF0', '#FE6100', '#29CA6E', '#DC267F', '#FFB000', '#34495E']


custom_styles = {
    'image.cmap': 'turbo',
    'figure.figsize': (24, 16),
    'text.color': '#212121',
    'axes.titleweight': 'bold',
    'axes.titlesize': 80,
    'axes.titlepad': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelpad': 12,
    'axes.labelcolor': '#212121',
    'axes.labelweight': 600,
    'axes.linewidth': 4,
    'axes.edgecolor': '#212121',
    'grid.linewidth': 4,
    'xtick.major.pad': 12,
    'ytick.major.pad': 12,
    'lines.linewidth': 10,
    'axes.labelsize': 80,
    'xtick.labelsize': 72,
    'ytick.labelsize': 72,
    'legend.fontsize': 64,
}


def list_img_files(dir):
    return [os.path.join(dir, fn) for fn in natsorted(os.listdir(dir), key=lambda y: y.lower()) if imghdr.what(os.path.join(dir, fn)) in ['tiff', 'jpeg', 'png', 'gif']]


def setup_dirs(dirs):
    try:
        os.makedirs(dirs)
    except OSError:
        pass


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
