import os


CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
MM_CFG_FILE = os.path.join(CONFIG_DIR, 'mm_nikon2.cfg')


SETTINGS = {
    'img_sync': ['XYStage', 'TIZDrive', 'Wheel-A', 'Wheel-B', 'Wheel-C', 'TIFilterBlock1'],
    'ch_group': 'Channel',
    'obj_device': 'TINosePiece',
    'xy_device': 'XYStage',
    'z_device': 'TIZDrive',
    'img_width_px': 1200,
    'img_width_um': 132,
    'img_height_px': 1200,
    'img_height_um': 132,
    'pixel_size': 0.11,
    'stage_z_limit': [-240, 240],
    'stage_x_limit': [-9600, 9600],
    'stage_y_limit': [-9600, 9600],
    'mpos': False,
    'mpos_ch': 'mCherry',
    'mpos_mode': 'sequential',  # "sequential" or "parallel"
}


## Seconds-Timescale ##
# IMAGING = {
#     't_info': [(0, 56, 5), (60, 62, 1), (65, 301, 5)],  # (start, stop, period)
#     'chs': ['mCherry']
# }

# INDUCTION = {
#     't_info': [(60, 61, 1, 1)],  # (start, stop, period, width)
#     'ch_ind': 'BL'
# }


## Seconds-Timescale ## Pulsatile ##
# IMAGING = {
#     't_info': [(0, 301, 5)],  # (start, stop, period)
#     'chs': ['mCherry']
# }

# INDUCTION = {
#     't_info': [(60, 181, 20, 1), (183, 241, 3, 1)],  # (start, stop, period, width)
#     'ch_ind': 'BL'
# }


## Minutes-Timescale ##
IMAGING = {
    't_info': [(0, 301, 5)],
    'chs': ['mCherry']
}

INDUCTION = {
    't_info': [(60, 300, 15, 0.1)],
    'ch_ind': 'BL10x'
}


# Hours-Timescale ## 
# IMAGING = {
#     't_info': [(0, 57601, 300)],
#     'chs': ['mCherry', 'DIC']
# }

# INDUCTION = {
#     't_info': [(0, 57601, 30, 1)],
#     'ch_ind': 'BL10x'
# }

AUTOFOCUS = {
    # 't_info': [(0, 57601, 300)],
    # 'ch': 'DIC',
    # 'bounds': [-2.0, 2.0],
    # 'z_step': 0.5,
    # 'offset': 0
}