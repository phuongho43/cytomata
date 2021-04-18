import os


CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
MM_CFG_FILE = os.path.join(CONFIG_DIR, 'mm_nikon2.cfg')


SETTINGS = {
    'img_sync': ['XYStage', 'TIZDrive', 'Wheel-A', 'Wheel-B', 'Wheel-C', 'TIFilterBlock1'],
    'cam_device': 'Prime95B',
    'ch_group': 'Channel',
    'obj_device': 'TINosePiece',
    'xy_device': 'XYStage',
    'z_device': 'TIZDrive',
    'stage_z_limit': [-240, 240],
    'stage_x_limit': [-9600, 9600],
    'stage_y_limit': [-9600, 9600],
    'mpos': False,
    'mpos_ch': 'mCherry',
    'mpos_mode': 'sequential',  # "sequential" or "parallel"
}


## Seconds-Timescale ##
# IMAGING = {
#     't_info': [(0, 56, 5), (60, 62, 1), (65, 181, 5)],  # (start, stop, period)
#     'chs': ['mCherry']
# }

# INDUCTION = {
#     't_info': [(60, 61, 1, 1)],  # (start, stop, period, width)
#     'ch_ind': 'BL100'
# }


## Seconds-Timescale ## Pulsatile ##
# IMAGING = {
#     't_info': [(0, 301, 5)],  # (start, stop, period)
#     'chs': ['mCherry']
# }

# INDUCTION = {
#     't_info': [(60, 181, 5, 1)],  # (start, stop, period, width)
#     'ch_ind': 'BL10'
# }


# Minutes-Timescale ##
# IMAGING = {
#     't_info': [(0, 301, 5)],
#     'chs': ['mCherry']
# }

# INDUCTION = {
#     't_info': [(60, 300, 5, 1)],
#     'ch_ind': 'BL10'
# }


# Hours-Timescale ## 
IMAGING = {
    't_info': [(0, 43201, 300)],
    'chs': ['YFP', 'mCherry']
}

INDUCTION = {
    't_info': [(0, 43200, 10, 1)],
    'ch_ind': 'BL10'
}

AUTOFOCUS = {
    't_info': [(0, 43200, 300)],
    'ch': 'DIC',
    'bounds': [-3.0, 3.0],
    'z_step': 1,
    'offset': 0
}