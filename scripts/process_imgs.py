import os
import sys
import warnings
sys.path.append(os.path.abspath('../'))

import numpy as np
from tqdm import tqdm
from imageio import mimwrite
from skimage import img_as_float
from skimage.io import imread

from cytomata.plot import plot_cell_img, plot_bkg_profile, plot_uy
from cytomata.process import preprocess_img, segment_object, segment_clusters, process_u_csv
from cytomata.utils import setup_dirs, list_img_files, rescale, custom_styles, custom_palette

def iter_cb(img, prog):
    return False

def process_fluo_timelapse(img_dir, save_dir, u_csv=None,
    t_unit='s', ulabel='BL', sb_microns=22, cmax=None,
    segmt=False, segmt_dots=False, segmt_mask=None, segmt_factor=1,
    remove_small=None, fill_holes=None, clear_border=None, adj_bright=False, iter_cb=iter_cb):
    """Analyze fluorescence timelapse images and generate figures."""
    if cmax is None:
        cmax = np.max([np.percentile(img_as_float(imread(imgf)), 99.9) for imgf in list_img_files(img_dir)])
    n_imgs = len(list_img_files(img_dir))
    t = [np.float(os.path.splitext(os.path.basename(imgf))[0]) for imgf in list_img_files(img_dir)]
    y = []
    tu = []
    u = []
    t_ann_img = []
    imgs = []
    if os.path.isfile(u_csv) and os.path.exists(u_csv):
        tu, u, t_ann_img = process_u_csv(t, u_csv, save_dir)
    factor = segmt_factor
    for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        fname = str(round(float(fname), 2))
        img, raw, bkg, den = preprocess_img(imgf)
        plot_bkg_profile(fname, save_dir, raw, bkg)
        thr = None
        yi = np.mean(img)
        if segmt:
            if os.path.isfile(segmt_mask) and os.path.exists(segmt_mask):
                seg_bound = img_as_float(imread(segmt_mask)) > 0
            if adj_bright:
                a_reg = img[img > 0]
                if os.path.isfile(segmt_mask) and os.path.exists(segmt_mask):
                    a_reg = seg_bound*img
                    a_reg = a_reg[a_reg > 0]
                if i == 0:
                    kval = np.mean(a_reg)
                segmt_factor = (np.mean(a_reg)/kval) + factor - 1
            if segmt_dots:
                thr = segment_clusters(den, factor=segmt_factor, rs=remove_small)
            else:
                thr = segment_object(den, factor=segmt_factor,
                    rs=remove_small, fh=fill_holes, cb=clear_border)
            if os.path.isfile(segmt_mask) and os.path.exists(segmt_mask):
                thr *= seg_bound
            yi = np.mean(img[thr])
            if np.isnan(yi):
                yi = np.mean(img)
        y.append(yi)
        sig_ann = round(float(fname), 1) in t_ann_img
        cell_img = plot_cell_img(den, thr, fname, save_dir,
            cmax, sig_ann, t_unit=t_unit, sb_microns=sb_microns)
        imgs.append(cell_img)
        prog = (i+1)/n_imgs * 100
        if iter_cb(cell_img, prog):
            break
    plot_uy(t, y, tu, u, save_dir, t_unit=t_unit, ulabel=ulabel)
    data = np.column_stack((t, y))
    np.savetxt(os.path.join(save_dir, 'y.csv'),
        data, delimiter=',', header='t,y', comments='')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mimwrite(os.path.join(save_dir, 'cell.gif'), imgs, fps=len(imgs)//10)


def process_fluo_images(img_dir, save_dir,
    sb_microns=22, cmax=None, segmt=False, segmt_dots=False,
    segmt_mask_dir='', segmt_factor=1, remove_small=None,
    fill_holes=None, clear_border=None, iter_cb=iter_cb):
    """Analyze fluorescence 10x images and generate figures."""
    if cmax is None:
        cmax = np.max([np.percentile(img_as_float(imread(imgf)), 99.9) for imgf in list_img_files(img_dir)])
    n_imgs = len(list_img_files(img_dir))
    y = []
    imgs = []
    for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        img, raw, bkg, den = preprocess_img(imgf)
        plot_bkg_profile(fname, save_dir, raw, bkg)
        thr = None
        yi = np.mean(img)
        if segmt:
            segmt_mask = ''
            if segmt_mask_dir is not None:
                segmt_mask = os.path.join(segmt_mask_dir, fname)
            if os.path.isfile(segmt_mask) and os.path.exists(segmt_mask):
                seg_bound = img_as_float(imread(segmt_mask)) > 0
            if segmt_dots:
                thr = segment_clusters(den, factor=segmt_factor, rs=remove_small)
            else:
                thr = segment_object(den, factor=segmt_factor,
                    rs=remove_small, fh=fill_holes, cb=clear_border)
            if os.path.isfile(segmt_mask) and os.path.exists(segmt_mask):
                thr *= seg_bound
            yi = np.mean(img[thr])
            if np.isnan(yi):
                yi = np.mean(img)
        y.append(yi)
        cell_img = plot_cell_img(den, thr, fname, save_dir,
            cmax, sig_ann=False, t_unit=None, sb_microns=sb_microns)
        prog = (i+1)/n_imgs * 100
        if iter_cb(cell_img, prog):
            break
    np.savetxt(os.path.join(save_dir, 'y.csv'),
        np.array(y), delimiter=',', header='y', comments='')


if __name__ == '__main__':
    # i = 0
    # root_dir = '/home/phuong/data/FPs/p53Tetra/20210109_CIB-mTq2-p53TetraV1_CRY2-mCh/'
    # save_dir = os.path.join(root_dir, 'results', str(i))
    # img_dir = os.path.join(root_dir, 'mCherry', str(i))
    # u_csv = os.path.join(root_dir, 'u{}.csv'.format(i))
    # segmt = os.path.join(root_dir, 'mask.tif')
    # process_fluo_timelapse(img_dir, save_dir, u_csv=u_csv,
    #     t_unit='s', ulabel='BL', sb_microns=22, cmax=None,
    #     segmt=False, segmt_dots=False, segmt_mask=segmt, segmt_factor=2,
    #     remove_small=2000, fill_holes=None, clear_border=None, adj_bright=False)
    
    root_dir = '/home/phuong/data/FPs/p53Tetra/20210129_CIBN-mTq2-p53TetraV1/'
    img_dir = os.path.join(root_dir, 'Default')
    save_dir = os.path.join(root_dir, 'results')
    process_fluo_images(img_dir, save_dir,
        sb_microns=22, cmax=None, segmt=False, segmt_dots=False, segmt_mask_dir='',
        segmt_factor=1, remove_small=15, fill_holes=None, clear_border=None)
