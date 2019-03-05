import os
import sys
import math
import time
import json
from datetime import datetime as dt, timedelta as td

import cv2
import shutil
import numpy as np
import exifread as er
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# Binary search function.
def bin_search(array, x):
    lo, hi = 0, len(array) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if array[mid] < x:
            lo = mid + 1
        elif x < array[mid]:
            hi = mid - 1
        else:
            return True
    return False


def process_img(image):
    cropped = image[:979, :]
    cropped[882:979, :203] = cropped[785:882, :203]
    equalized = cv2.equalizeHist(cropped)
    return equalized


def get_median(image, PIXEL):
    hist = cv2.calcHist([image], [0], None, [256], [0,256])

    cumulative = 0
    for i, count in enumerate(hist):
        cumulative += count
        if cumulative > PIXEL / 2:
            median = i
            break

    return median


def dt_ISO(datetime):
    return dt.strftime(datetime, "%Y%m%dT%H%M%S")


def td_minutes(td):
    return round(td.days * 1440 + td.seconds / 60, 2)


''' Empty class used for variables modified at local scopes,
but accessed elsewhere.
'''
class Mem():
    def __init__(self):
        pass


# Class Cam is the meat of this code. Far too bulky, needs to be broken up.
class Cam():
    def __init__(self):
        root = tk.Tk()

        mem.DISP_WIDTH = root.winfo_screenwidth()

        root.withdraw()
        ROOT_PATH = filedialog.askdirectory()
        root.update()
        root.destroy()


        self.ROOT_PATH = ROOT_PATH
        self.event_thresh = 20000
        self.mean_diff = 0.9
        self.smooth = 1
        self.night = 0

    ''' The "whole shabang." Load if possible, otherwise process,
    then plot and save.
    '''
    def shabang(self):
        print('INITIALIZED')
        if self.load() is not None:
            print('PRE-PROCESSED')
        else:
            self.process()
            print('PROCESSED')

        self.plot()
        self.finish()
        print('FINISHED')
        print(f" ∙ There should be {sum(self.bools)} images exported.")

    # Populate Cam object with dates, times, names, paths, dts, etc.
    def process(self):
        self.datetimes, self.names, self.paths = self.get_EXIF()
        print(" ∙ EXIF read.")

        self.deltas = self.get_deltas()

        mem.temp = (self.datetimes[i].strftime("%H %M").split() for
                    i in range(len(self.names)))
        mem.temp = (int(x) + int(y) / 60 for x, y in mem.temp)
        self.hours = [min(abs(x-1), abs(x-25)) for x in mem.temp]

        self.counts, self.means = self.get_counts()
        print(" ∙ Images processed.")
        self.new_counts = self.get_new_counts()
        self.save()

    # Save selected images in folder.
    def finish(self):
        self.save()
        self.bools = self.get_events()
        self.runs = self.get_runs()

        folder = (self.event_thresh, self.mean_diff, self.smooth, self.night)
        folder_str = dt_ISO(dt.now()) + '_' + '_'.join(
            map(lambda x: str(round(x, 1)), folder))

        selected = os.path.join(self.ROOT_PATH, '_selected', folder_str)
        if not os.path.exists(selected):
            os.makedirs(selected)

        pad = len(str(max(self.runs)))
        for i, boo in enumerate(self.bools):
            if boo:
                shutil.copy2(self.paths[i], os.path.join(
                    selected, (f"{str(self.runs[i]).rjust(pad, '0')}_"
                               f"{dt_ISO(self.datetimes[i])}_"
                               f"{self.names[i]}")))

    ''' Save data read from EXIF, plot settings, and selected images
    as JSON objects.
    '''
    def save(self):
        saves = os.path.join(self.ROOT_PATH, '.saves')
        if not os.path.exists(saves):
            os.makedirs(saves)

        sub_paths = list(map(lambda x: x.replace(self.ROOT_PATH + '/', ''),
                             self.paths))
        date_strings = list(map(lambda x: x.__str__(), self.datetimes))
        self.jpg_data = list(zip(sub_paths, date_strings,
                                 self.counts, self.means))

        temp = [self.event_thresh, self.mean_diff, self.smooth, self.night]
        with open(os.path.join(saves, 'plot_params.dat'), 'w') as f:
            f.write(json.dumps(temp))

        with open(os.path.join(saves, 'jpg_data.dat'), 'w') as f:
            f.write(json.dumps(self.jpg_data))

        with open(os.path.join(saves, 'bumps.dat'), 'w') as f:
            f.write(json.dumps(mem.bumps))

    # Load saved JSON objects.
    def load(self):
        saves = os.path.join(self.ROOT_PATH, '.saves')
        if os.path.exists(saves):

            with open(os.path.join(saves, 'plot_params.dat'), 'r') as f:
                text = f.read()
                self.event_thresh, self.mean_diff, self.smooth, self.night = (
                    json.loads(text)
                )

            with open(os.path.join(saves, 'jpg_data.dat'), 'r') as f:
                text = f.read()
                self.jpg_data = json.loads(text)

            with open(os.path.join(saves, 'bumps.dat'), 'r') as f:
                text = f.read()
                temp_dict = json.loads(text)
                mem.bumps = {int(k): v for k, v in temp_dict.items()}

            sub_paths, date_strings, self.counts, self.means = (
                list(zip(*self.jpg_data)))
            self.paths = list(map(lambda x: os.path.join(self.ROOT_PATH, x),
                                  sub_paths))
            self.names = [x.split('/')[-1] for x in self.paths]
            self.datetimes = [dt.strptime(x, '%Y-%m-%d %H:%M:%S') for
                              x in date_strings]
            self.deltas = self.get_deltas()

            mem.temp = (self.datetimes[i].strftime("%H %M").split() for
                        i in range(len(self.names)))
            mem.temp = (int(x) + int(y) / 60 for x, y in mem.temp)
            self.hours = [min(abs(x-1), abs(x-25)) for x in mem.temp]

            self.new_counts = self.get_new_counts()
            return True
        else:
            return None

    # Read EXIF data from JPGs, take datetime.
    def get_EXIF(self):
        output = []

        for dir_, subdir, files in os.walk(self.ROOT_PATH):
            if '_selected' not in dir_:
                found = (f for f in files if f.endswith('.JPG'))
                for jpg in found:
                    filepath = os.path.join(dir_, jpg)
                    file = open(filepath, "rb")
                
                    tags = er.process_file(file,
                                           details = False,
                                           stop_tag = "Image DateTime")

                    datetime = dt.strptime(
                        str(tags["Image DateTime"]), "%Y:%m:%d %H:%M:%S"
                    )
                    tup = (datetime, jpg, filepath)
                    output.append(tup)

        return list(zip(*sorted(output)))

    # Find timedeltas in between successive images.
    def get_deltas(self):
        return [self.datetimes[i] - self.datetimes[i-1] if
                i > 0 else td(0) for
                i in range(len(self.datetimes))]

    # Process images using frame differencing techniques.
    def get_counts(self):
        approx = dt.now() + td(seconds=len(self.names) / 12)
        print(" ∙ Image processing started, come back at "
              f"{approx.strftime('%H:%M:%S')}.")
        
        output_counts = []
        output_medians = []

        prev = process_img(cv2.imread(self.paths[0], 0))
        w, h = prev.shape
        PX_COUNT = w * h

        for path in self.paths:
            img = cv2.imread(path, 0)
            MEDIAN = get_median(img, PX_COUNT)
            
            curr = process_img(img)
            difference = cv2.absdiff(curr, prev)
            blurred = cv2.medianBlur(difference, 11)
            _, mask = cv2.threshold(blurred,
                                    MEDIAN*1.05, 255,
                                    cv2.THRESH_BINARY)

            _, contours, _ = cv2.findContours(
                mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )

            count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    count += area

            output_counts.append(count)
            output_medians.append(MEDIAN)
            prev = curr.copy()

        return output_counts, output_medians

    ''' Modify counts to eliminate day to night transitions,
    multiply up values occuring at night.
    '''
    def get_new_counts(self):
        output = []

        for i, count in enumerate(self.counts):
            if i == 0:
                count = self.counts[i+1]
            elif i == len(self.counts) - 1:
                pass
            else:
                mean_new, mean_old = self.means[i], self.means[i-1]
                if (abs(mean_new - mean_old) /
                    min(mean_new, mean_old)) > self.mean_diff:
                    count = -1
                count *= 1 + (self.night / (1 + 150**(self.hours[i]-4.5)))

            output.append(count)

        for i, shift in mem.bumps.items():
            output[i] = shift

        return output

    ''' Creates events (images above threshold, and their neighbors
    consecutive in time) and trims them.
    '''
    def get_events(self):
        output = [False] * len(self.new_counts)
        len_output = len(output)

        for i in range(len_output):
            if i == 0:
                check = self.new_counts[i+1] > self.event_thresh
            elif i == len_output - 1:
                check = self.new_counts[i] > self.event_thresh
            else:
                check = (self.new_counts[i] > self.event_thresh or
                         self.new_counts[i+1] > self.event_thresh)

            if check:
                output[i] = True

        prohibited = list(mem.bumps.keys())
        prohibited += [x-len_output-1 for x in prohibited]
        prohibited.sort()

        for i, move in ((0, 1), (-1, -1)):
            while i in range(-len_output, len_output):
                if (i != 0 and i != len_output - 1 and not output[i] and
                    not bin_search(prohibited, i)):
                    if (move == 1 and output[i-1] > 0 and
                        td_minutes(self.deltas[i]) < self.smooth):
                        output[i] = True
                    elif (move == -1 and output[i+1] > 0 and
                          td_minutes(self.deltas[i+1]) < self.smooth):
                        output[i] = True
                i += move

        return output

    # Labels events with a "run" number, which is an unbroken streak.
    def get_runs(self):
        output = [0] * len(self.bools)

        num = 1
        prev_dt = dt(2000, 1, 1)
        for i, boo in enumerate(self.bools):
            curr_dt = self.datetimes[i]
            if boo:
                delta = curr_dt - prev_dt
                if td_minutes(delta) > 5:
                    num += 1
                output[i] = num
            prev_dt = curr_dt

        return output

    ''' Meaty interactive plot whereby a user sees plotted response values
    and can modify and edit the selection.
    '''
    def plot(self):
        def update():
            mem.red.remove()
            mem.black.remove()
            mem.cyan.remove()
            mem.grey.remove()

            lenn = len(self.new_counts)
            mem.grey = ax.fill_between(
                np.arange(0, len(self.new_counts)) + 0.5, 0, 100000,
                where=mem.bools,
                facecolor='#D8BFAA', alpha=0.5)
            mem.red = ax.fill_between(
                np.arange(0, lenn) + 0.5, 0, 100000,
                where=self.bools, facecolor='#F00314')
            mem.black = ax.fill_between(
                range(lenn), 0, self.new_counts,
                facecolor = 'black')
            mem.cyan = ax.axhline(self.event_thresh, color='#14B37D')


            fig.canvas.draw_idle()

        def on_slide(val):
            if (self.mean_diff != diff_slider.val or
                self.night != night_slider.val):
                self.mean_diff = diff_slider.val
                self.night = night_slider.val

                self.new_counts = self.get_new_counts()
                self.bools = self.get_events()
                mem_bools()
            else:
                self.event_thresh = event_slider.val
                self.smooth = smooth_slider.val

                self.bools = self.get_events()
            update()

        def image_pano(xdata):
            xdat = int(round(xdata, 0)) - 1
            len_ = len(self.new_counts)
            array = list(range(xdat - 1, xdat + 3))
            while not all(a in range(0, len_) for a in array):
                if any(a < 0 for a in array):
                    array = list(map(lambda x: x + 1, array))
                elif any(a >= len_ for a in array):
                    array = list(map(lambda x: x - 1, array))

            stack = []
            for n in array:
                if n in mem.images_keys:
                    ind = mem.images_keys.index(n)
                    img = mem.images[ind]
                    mem.images_keys.pop(ind)
                    mem.images_keys.append(n)
                    mem.images.pop(ind)
                    mem.images.append(img)
                else:
                    img = cv2.imread(self.paths[n])
                    mem.images_keys = mem.images_keys[1:] + [n]
                    mem.images = mem.images[1:] + [img]

                if mem.switch:
                    img = cv2.cvtColor(img[:980, :], cv.COLOR_BGR2GRAY)
                    img = cv2.equalizeHist(img)
                
                stack.append(img)

            pano = np.hstack(stack)
            h, w, *_ = pano.shape
            scale = w / mem.DISP_WIDTH
            cv2.namedWindow("Main", cv2.WINDOW_NORMAL)
            cv2.imshow("Main",
                       cv2.resize(pano, (w // 2, h // 2))
                       )

        def on_click(event):
            if event.dblclick and event.xdata is not None:
                image_pano(event.xdata)

        def on_key(event):
            first = mem.bum[0] is None
            if event.xdata is not None:
                i = int(round(event.xdata))
                if first:
                    if event.key == 'z':
                        mem.bum = [-1, i]
                        image_pano(event.xdata)
                    elif event.key == 'x':
                        mem.bum = [1e5, i]
                        image_pano(event.xdata)
                    elif event.key == ',':
                        mem.bum = [0, i]
                        image_pano(event.xdata)
                if event.key in 'zxc,':
                    image_pano(event.xdata)
                elif event.key == '`':
                    if mem.switch:
                        mem.switch = False
                    else:
                        mem.switch = True
                    image_pano(event.xdata)
                elif event.key == '.':
                    mem.bumps = {}

                    self.new_counts = self.get_new_counts()
                    self.bools = self.get_events()
                    mem_bools()

                    update()

        def off_key(event):
            try:
                if event.xdata is not None:
                    low, high = sorted((mem.bum[1], int(round(event.xdata))))
                    if event.key == 'z' or event.key == 'x':
                        mem.bumps = {**mem.bumps,
                                     **{i: mem.bum[0] for i in
                                        range(max(0, low),
                                              min(len(self.counts), high+1))}
                            }
                    elif event.key == ',':
                        for i in range(max(0, low),
                                       min(len(self.counts), high+1)):
                            mem.bumps.pop(i, None)
                    mem.bum = [None, None]
                elif event.key == ',' and event.xdata is not None:
                    low, high = sorted((mem.bum[1], int(round(event.xdata))))

                self.new_counts = self.get_new_counts()
                self.bools = self.get_events()
                mem_bools()

                update()
            except:
                pass

        def mem_bools():
            mem.bools = [self.new_counts[i] < 0 or
                         self.new_counts[i+1] < 0 for
                         i in range(len(self.new_counts)-1)] + [True]           

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(0.07, 0.20, 0.97, 0.97)
        fig.canvas.set_window_title('Response Filtering')

        font = {'size': 8}
        plt.rc('font', **font)

        ax.grid(alpha = 0.4)
        ax.set_axisbelow(True)
        ax.set_ylim([0, 40000])
        ax.set_xlim([0, min(len(self.new_counts), 500)])
        ax.set_xlabel("Frame")
        ax.set_ylabel("Response")
        plt.yticks(rotation=45)

        plt.rcParams['keymap.back'] = 'left, backspace'

        mem.red = ax.axhline()
        mem.cyan = ax.axhline()
        mem.black = ax.axhline()
        mem.grey = ax.axhline()

        event_slider_ax  = fig.add_axes([0.25, 0.08, 0.6, 0.02])
        event_slider = Slider(event_slider_ax, 'EVENT', 0, 40000,
                              valinit=self.event_thresh, valfmt = '%i',
                              color='#003459', alpha=0.5)
        diff_slider_ax = fig.add_axes([0.25, 0.06, 0.6, 0.02])
        diff_slider = Slider(diff_slider_ax, 'MEAN DIFF', 0, 2,
                              valinit=self.mean_diff, valfmt = '%1.1f',
                              color='#003459', alpha=0.5)
        smooth_slider_ax = fig.add_axes([0.25, 0.04, 0.6, 0.02])
        smooth_slider = Slider(smooth_slider_ax, 'SMOOTH', 0, 5,
                               valinit=self.smooth, valfmt = '%1.1f',
                              color='#003459', alpha=0.5)
        night_slider_ax  = fig.add_axes([0.25, 0.02, 0.6, 0.02])
        night_slider = Slider(night_slider_ax, 'NIGHT', 0, 50,
                              valinit=self.night, valfmt = '%i',
                              color='#003459', alpha=0.5)

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('key_release_event', off_key)

        event_slider.on_changed(on_slide)
        diff_slider.on_changed(on_slide)
        smooth_slider.on_changed(on_slide)
        night_slider.on_changed(on_slide)

        ax.fill_between(np.arange(0, len(self.hours)) + 0.5, 0, 100000,
                        where=[x < 4.5 for x in self.hours],
                        facecolor='#003459', alpha=0.5)

        self.new_counts = self.get_new_counts()
        self.bools = self.get_events()
        mem_bools()

        update()
        plt.show()
        cv2.destroyAllWindows()

mem = Mem()
mem.bumps = {}
mem.bum = [None, None]
mem.images_keys = [-1] * 64
mem.images = [''] * 64
mem.switch = False

cam = Cam()
cam.shabang()
