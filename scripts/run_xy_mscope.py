import os
import sys
import time
import shutil
sys.path.append(os.path.abspath('../'))

from cytomata.utils import setup_dirs
from cytomata.mscope import Microscope
from configs.mm_settings import CONFIG_DIR, MM_CFG_FILE, SETTINGS, INDUCTION, IMAGING, AUTOFOCUS


expt_name = input('Expt Directory Name: ')
expt_name = ''.join([x if x.isalnum() or x in '-_.' else '_' for x in expt_name])
SETTINGS['save_dir'] = os.path.join('expts', time.strftime('%Y%m%d_') + expt_name)
if os.path.exists(SETTINGS['save_dir']):
    input('Directory already exists. Overwrite?...Press Ctrl-C to abort. Press ENTER to continue.')
setup_dirs(SETTINGS['save_dir'])
settings_file = os.path.join(CONFIG_DIR, 'mm_settings.py')
settings_file_save = os.path.join(SETTINGS['save_dir'], 'settings.txt')
shutil.copyfile(settings_file, settings_file_save)
cfg_file_save = os.path.join(SETTINGS['save_dir'], 'configs.txt')
shutil.copyfile(MM_CFG_FILE, cfg_file_save)


mscope = Microscope(SETTINGS, MM_CFG_FILE)
mscope.core.clear_roi()
exp0 = mscope.core.get_exposure()
mscope.core.set_exposure(0)
mscope.set_channel('mCherry')
mscope.core.set_auto_shutter(False)
mscope.core.set_shutter_open(False)
img = mscope.snap_image()
mscope.core.set_exposure(exp0)
mscope.core.set_auto_shutter(True)


mscope.snap_xyfield(chs=['mCherry'], n=3, step=1320)