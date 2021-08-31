import os


CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
MM_CFG_FILE = os.path.join(CONFIG_DIR, 'mm_nikon2.cfg')
IMAGING = {}
INDUCTION = {}
AUTOFOCUS = {}


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
    'mpos': '', # '', 'sequential', 'parallel'
    'mpos_ch': 'mCherry',
    'roi_center': False,
}


# Seconds-Timescale ##
IMAGING = {
    't_info': [(0, 56, 5), (60, 62, 1), (65, 181, 5)],  # (start, stop, period)
    'chs': ['mCherry']
}

INDUCTION = {
    't_info': [(60, 61, 1, 1)],  # (start, stop, period, width)
    'ch_ind': 'BL1'
}


# Seconds-Timescale ## Pulsatile ##
# IMAGING = {
#     't_info': [(0, 181, 5)],  # (start, stop, period)
#     'chs': ['mCherry']
# }

# INDUCTION = {
#     't_info': [(60, 91, 3, 1)],  # (start, stop, period, width)
#     'ch_ind': 'BL1'
# }


# Minutes-Timescale ##
# IMAGING = {
#     't_info': [(0, 301, 10)],
#     'chs': ['mCherry', 'CFP']
# }

# INDUCTION = {
#     't_info': [(60, 300, 10, 8)],
#     'ch_ind': 'BL1'
# }


# Hours-Timescale ## 
# IMAGING = {
#     't_info': [(0, 57601, 300)],
#     'chs': ['DIC', 'TxRed']
# }

# INDUCTION = {
#     't_info': [(0, 57600, 10, 1)],
#     'ch_ind': 'BL1'
# }

# AUTOFOCUS = {
#     't_info': [(0, 57600, 300)],
#     'ch': 'DIC',
#     'bounds': [-20.0, 20.0],
#     'z_step': 10.0,
#     'offset': 5
# }