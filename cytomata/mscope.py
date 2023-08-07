import os
import time
import warnings
from collections import defaultdict, deque

import numpy as np
import cv2
from skimage.io import imsave
from skimage.filters import laplace

from pycromanager import Bridge
from cytomata.utils import setup_dirs, clear_screen


class Microscope(object):
    """Microscope task automation and data recording."""
    def __init__(self, settings, config_file):
        self.bridge = Bridge()
        self.core = self.bridge.get_core()
        self.core.load_system_configuration(config_file)
        self.tasks = []
        self.settings = settings
        self.save_dir = settings['save_dir']
        self.ch_group = self.settings['ch_group']
        self.obj_device = self.settings['obj_device']
        self.xy_device = self.settings['xy_device']
        self.z_device = self.settings['z_device']
        for dev in self.settings['img_sync']:
            self.core.assign_image_synchro(dev)
        self.uta = defaultdict(list)
        self.utb = defaultdict(list)
        self.x0, self.y0 = self.get_position('xy')
        self.z0 = self.get_position('z')
        self.xlim = np.array(settings['stage_x_limit']) + self.x0
        self.ylim = np.array(settings['stage_y_limit']) + self.y0
        self.zlim = np.array(settings['stage_z_limit']) + self.z0
        self.coords = np.array([[self.x0, self.y0, self.z0]])
        self.cid = 0
        self.t0 = time.time()

    def set_channel(self, chname):
        self.core.set_config(self.ch_group, chname)

    def set_magnification(self, mag):
        self.core.set_state(self.obj_device, mag)

    def get_position(self, axis):
        if axis.lower() == 'xy':
            x = self.core.get_x_position(self.xy_device)
            y = self.core.get_y_position(self.xy_device)
            return x, y
        elif axis.lower() == 'z':
            return self.core.get_position(self.z_device)
        else:
            raise ValueError('Invalid axis arg in mscope.get_position(axis).')

    def set_position(self, axis, value):
        if axis.lower() == 'xy':
            if (value[0] > self.xlim[0] and value[0] < self.xlim[1] and value[1] > self.ylim[0] and value[1] < self.ylim[1]):
                self.core.set_xy_position(self.xy_device, round(value[0]), round(value[1]))
        elif axis.lower() == 'z':
            if value > self.zlim[0] and value < self.zlim[1]:
                self.core.set_position(self.z_device, value)
        else:
            raise ValueError('Invalid axis arg in mscope.set_position(axis).')

    def convert_tagged_img(self, tagged_img):
        img_h = tagged_img.tags['Height']
        img_w = tagged_img.tags['Width']
        return tagged_img.pix.reshape((img_h, img_w))

    def add_coord(self):
        x, y = self.get_position('xy')
        z = self.get_position('z')
        self.coords = np.vstack(([x, y, z], self.coords))

    def add_coords_session(self, ch):
        self.set_channel(ch)
        n_bits = self.core.get_image_bit_depth()
        cv2.namedWindow('Coordinate Picker')
        self.core.start_continuous_sequence_acquisition(1)
        while True:
            if self.core.get_remaining_image_count() > 0:
                timg = self.core.get_last_tagged_image()
                img = self.convert_tagged_img(timg)
                img = cv2.normalize(img, dst=None, alpha=0,
                    beta=2**n_bits - 1, norm_type=cv2.NORM_MINMAX)
                cv2.imshow('Coordinate Picker', img)
            k = cv2.waitKey(20)
            if k == 27:  # ESC - Exit
                break
            elif k == 32:  # SPACE - Add Current Coord
                self.add_coord()
                print('--Coords List--')
                for coord in self.coords:
                    print(coord)
            elif k == 8:  # Backspace - Remove Prev Coord
                self.coords = self.coords[1:]
                print('--Coords List--')
                for coord in self.coords:
                    print(coord)
        cv2.destroyAllWindows()
        self.core.stop_sequence_acquisition()
        clear_screen()

    def snap_image(self):
        self.core.wait_for_system()
        self.core.clear_circular_buffer()
        self.core.snap_image()
        timg = self.core.get_tagged_image()
        img = self.convert_tagged_img(timg)
        return img

    def snap_zstack(self, chs, zdepth, step):
        z0 = self.get_position('z')
        zs = np.arange(z0 - zdepth/2, z0 + zdepth/2, step)
        tstamp = time.strftime('%Y%m%d-%H%M%S')
        img_dir = os.path.join(self.save_dir, tstamp + '_zstack')
        imgs = []
        for ch in chs:
            self.set_channel(ch)
            for z in zs:
                self.set_position('z', z)
                img = self.snap_image()
                imgs.append((z, ch, img))
        for z, ch, img in imgs:
            ch_dir = os.path.join(img_dir, ch)
            setup_dirs(ch_dir)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(os.path.join(ch_dir, str(z) + '.tiff'), img)
        self.set_position('z', z0)

    def snap_xyfield(self, chs, n=3, step=132):
        x0, y0 = self.get_position('xy')
        grid = np.arange(-(n//2)*step, (n//2)*step + step, step)
        imgs = []
        for i, yi in enumerate(grid):
            for j, xi in enumerate(grid):
                if not i % 2:
                    xi = -xi
                self.set_position('xy', (x0 + xi, y0 + yi))
                for ch in chs:
                    self.set_channel(ch)
                    img = self.snap_image()
                    imgs.append((i, j, ch, img))
        for i, j, ch, img in imgs:
            ch_dir = os.path.join(self.save_dir, ch)
            setup_dirs(ch_dir)
            img_path = os.path.join(ch_dir, str(i) + '_' + str(j) + '.tiff')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(img_path, img)
        self.set_position('xy', (x0, y0))

    def take_images(self, cid, chs):
        ti = time.time() - self.t0
        for ch in chs:
            self.set_channel(ch)
            img = self.snap_image()
            img_path = os.path.join(self.save_dir, ch, str(cid), str(round(ti, 1)) + '.tiff')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(img_path, img)

    def imaging_task(self, chs):
        if self.settings['mpos'] == 'parallel':
            for i in range(len(self.coords)):
                (x, y, z) = self.coords[i]
                self.set_position('xy', (x, y))
                self.take_images(i, chs)
        else:
            (x, y, z) = self.coords[self.cid]
            self.set_position('xy', (x, y))
            self.take_images(self.cid, chs)

    def queue_imaging(self, t_info, chs):
        for (start, stop, period) in t_info:
            self.tasks.append({
                'func': self.imaging_task,
                'times': deque(np.arange(start, stop, period)),
                'kwargs': {'chs': chs}
            })
        for ch in chs:
            for i in range(len(self.coords)):
                setup_dirs(os.path.join(self.save_dir, ch, str(i)))

    def pulse_light(self, cid, width, ch_ind):
        self.set_channel(ch_ind)
        self.core.set_auto_shutter(False)
        self.core.set_shutter_open(True)
        ta = time.time() - self.t0
        time.sleep(width)
        tb = time.time() - self.t0
        self.core.set_shutter_open(False)
        self.core.set_auto_shutter(True)
        self.uta[cid].append(ta)
        self.utb[cid].append(tb)
        u_path = os.path.join(self.save_dir, 'u' + str(cid) + '.csv')
        u_data = np.column_stack((self.uta[cid], self.utb[cid]))
        np.savetxt(u_path, u_data, delimiter=',', header='ta,tb', comments='')

    def induction_task(self, width, ch_ind):
        if self.settings['mpos'] == 'parallel':
            for i in range(len(self.coords)):
                (x, y, z) = self.coords[i]
                self.set_position('xy', (x, y))
                self.pulse_light(i, width, ch_ind)
        else:
            (x, y, z) = self.coords[self.cid]
            self.set_position('xy', (x, y))
            self.pulse_light(self.cid, width, ch_ind)

    def queue_induction(self, t_info, ch_ind):
        for (start, stop, period, width) in t_info:
            self.tasks.append({
                'func': self.induction_task,
                'times': deque(np.arange(start, stop, period)),
                'kwargs': {'width': width, 'ch_ind': ch_ind}
            })

    def measure_focus(self):
        img = self.snap_image()
        return np.var(laplace(img))

    def autofocus(self, cid, ch, bounds=[-10.0, 10.0], z_step=5.0, offset=0.0):
        self.set_channel(ch)
        zi = self.get_position('z')
        zl = zi + bounds[0]
        zu = zi + bounds[1]
        best_foc = 0
        best_pos = zi
        for zz in np.arange(zl, zu, z_step):
            self.set_position('z', zz)
            foc = self.measure_focus()
            if foc > best_foc:
                best_foc = foc
                best_pos = zz
        best_pos += offset
        self.coords[cid, 2] = best_pos
        self.set_position('z', best_pos)

    def autofocus_task(self, ch, bounds, z_step, offset):
        if self.settings['mpos'] == 'parallel':
            for i in range(len(self.coords)):
                (x, y, z) = self.coords[i]
                self.set_position('xy', (x, y))
                self.autofocus(i, ch, bounds, z_step, offset)
        else:
            (x, y, z) = self.coords[self.cid]
            self.set_position('xy', (x, y))
            self.autofocus(self.cid, ch, bounds, z_step, offset)

    def queue_autofocus(self, t_info, ch, bounds, z_step, offset):
        for (start, stop, period) in t_info:
            self.tasks.append({
                'func': self.autofocus_task,
                'times': deque(np.arange(start, stop, period)),
                'kwargs': {'ch': ch, 'bounds': bounds, 'z_step': z_step, 'offset': offset}
            })

    def run_tasks(self):
        if self.tasks:
            for i, task in enumerate(self.tasks):
                if task['times']:
                    if time.time() > task['times'][0] + self.t0:
                        task['func'](**task['kwargs'])
                        self.tasks[i]['times'].popleft()
                else:
                    self.tasks.pop(i)
        else:
            return True
