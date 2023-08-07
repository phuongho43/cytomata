import os
import imghdr
from natsort import natsorted


custom_palette = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#29CA6E', '#FFB000', '#34495E']


custom_styles = {
    'image.cmap': 'turbo',
    'figure.figsize': (16, 10),
    'text.color': '#212121',
    'axes.titleweight': 'bold',
    'axes.titlesize': 46,
    'axes.titlepad': 18,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelpad': 12,
    'axes.labelcolor': '#212121',
    'axes.labelweight': 600,
    'axes.linewidth': 3,
    'axes.edgecolor': '#212121',
    'xtick.major.pad': 8,
    'ytick.major.pad': 8,
    'lines.linewidth': 5,
    # 'axes.labelsize': 72,
    # 'xtick.labelsize': 64,
    # 'ytick.labelsize': 64,
    # 'legend.fontsize': 48,
    'axes.labelsize': 56,
    'xtick.labelsize': 48,
    'ytick.labelsize': 48,
    'legend.fontsize': 40,
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
