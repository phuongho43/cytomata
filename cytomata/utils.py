import os
import imghdr

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy.interpolate import interp1d
from skimage import img_as_ubyte
from cytomata import turbo_cmap


custom_palette = [
    '#1976D2', '#D32F2F', '#388E3C',
    '#7B1FA2', '#F57C00', '#C2185B',
    '#FBC02D', '#303F9F', '#0097A7',
    '#5D4037', '#455A64', '#AFB42B']


custom_styles = {
    'image.cmap': 'turbo',
    'figure.figsize': (16, 8),
    'text.color': '#212121',
    'axes.titleweight': 'bold',
    'axes.titlesize': 32,
    'axes.titlepad': 18,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 28,
    'axes.labelpad': 10,
    'axes.labelcolor': '#212121',
    'axes.labelweight': 600,
    'axes.linewidth': 3,
    'axes.edgecolor': '#212121',
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 28,
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
