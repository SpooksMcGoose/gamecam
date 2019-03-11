import os
import datetime
from datetime import datetime as dt, timedelta as td
from operator import itemgetter

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import exifread as er
import numpy as np
import cv2
import json


############################################################
###                       CONSTANTS                      ###
############################################################


DIRECTION_MULTS = {
    "A": (-1,-1,0,0),
    "B": (1,1,0,0),
    "R": (0,0,1,1),
    "L": (0,0,-1,-1)
}


############################################################
###                  GENERAL  FUNCTIONS                  ###
############################################################


# Consumes a generator to a list of the first N elements.
def first_n_elems(generator, n):
    output = []
    for _ in range(n):
        output.append(next(generator))
    return output


def to_24_hour(datetime):
    time = datetime.time()
    return time.hour + time.minute / 60 + time.second / 3600


def extract_var(data, var):
    return [x[var] for x in data]


############################################################
###                 SPECIFIC   FUNCTIONS                 ###
############################################################


# Recursive walk finding all JPG images below home_path.
def find_jpgs(home_path):
    output = []
    
    for dir_, subdir, files in os.walk(home_path):
        if '_selected' not in dir_:
            found = (
                f for f in files
                if f.upper().endswith(('.JPG', '.JPEG'))
            )
            for filename in found:                
                filepath = os.path.join(dir_, filename)

                output.append({
                    'filename': filename,
                    'filepath': filepath
                })

    return output


# Default parser for the EXIF "Image DateTime" tag.
def parse_dt(raw):
    return dt.strptime(str(raw), "%Y:%m:%d %H:%M:%S")


# Takes in filepaths, reads binary files, attaches EXIF data.
# Can create custom tuples in tag_format to extract more data.
def attach_exif(jpg_data, tag_format=[
    ("Image DateTime", 'datetime', parse_dt)
    ]):
    output = []

    for row in jpg_data:
        file = open(row['filepath'], "rb")
        tags = er.process_file(file, details = False)

        for t, var, anon in tag_format:
            row[var] = anon(tags[t])

        output.append(row)

    return output


# Quickly finds the median tone of a grayscale image.
def hist_median(image, px_count):
    hist = cv2.calcHist([image], [0], None, [256], [0,256])

    tally = 0
    threshold = px_count / 2
    for i, count in enumerate(hist):
        tally += count
        if tally > threshold:
            return i


# Takes single image. Crops / clones out timestamp, equalizes histogram.
# Importan pre-processing step for the process_jpgs function.
def process_img(image, crop, clone_params):
    h, w = image.shape

    image = image[crop[0]:h-crop[1], crop[2]:w-crop[3]]
    
    a, b, c, d, e, f, g, h = clone_params
    image[a:b, c:d] = image[e:f, g:h]

    return cv2.equalizeHist(image)


def generate_clone_params(clone_region, clone_from):
    clone_to = np.array(clone_region)
    mults = np.array(DIRECTION_MULTS[clone_from[0].upper()])

    a, b, c, d = clone_to
    h, w = b - a, d - c
    h_or_w = np.array([h, h, w, w])

    clone_from = clone_to + (h_or_w*mults)
    return list(clone_to) + list(clone_from)    


# Minimum input requires data with filepaths. Method combines frame-
# differencing, thresholding, and summation of contours to output response.
def process_jpgs(jpg_data, crop=(0,0,0,0), clone_params=(0,0,0,0,0,0,0,0)):
    output = []

    for i, row in enumerate(jpg_data):
        if i == 0:
            prev = process_img(cv2.imread(row['filepath'], 0),
                               crop, clone_params)
            h, w = prev.shape
            px_count = w * h

        jpg = cv2.imread(row['filepath'], 0)

        row['median'] = hist_median(jpg, px_count)
        curr = process_img(jpg, crop, clone_params)

        difference = cv2.absdiff(curr, prev)
        blurred = cv2.medianBlur(difference, 11)

        _, mask = cv2.threshold(blurred,
                                row['median']*1.05, 255,
                                cv2.THRESH_BINARY)

        _, contours, _ = cv2.findContours(
            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                count += area
        row['count'] = count

        '''
        cv2.imshow('uh', prev)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
        '''

        prev = curr
        output.append(row)

    return output


# Minimum input requires process_jpg data. Object essentially holds
# data used in exporting, parameters for graphing, and plot function.
class Cam():
    def __init__(self, jpg_data=False):
        if jpg_data:
            self.jpg_data = list(jpg_data)
            self.length = len(self.jpg_data)

            self.attach_diffs('datetime', 'timedelta')
            self.attach_diffs('median', 'med_diff')

            for row in self.jpg_data:
                hour = to_24_hour(row['datetime'])
                row['from_midnight'] = (
                    hour if hour < 12
                    else (hour - 24) * -1
                )
                row['td_minutes'] = round(
                    row['timedelta'].total_seconds() / 60, 2
                )
                row['med_diff'] = abs(row['med_diff'])

            self.plot_params = {
                "resp_thresh": 20000,
                "trans_thresh": 30,
                "smooth_time": 1,
                "night_mult": 0
            }

            self.lines = {}
            self.sliders = {}

            self.user_edits = {}
            self.press = [None, None]
            self.toggle = False

            self.buffer = [[None] * 64, [None] * 64]

    # ???
    def save(self, path=False):
        pass

    # ???
    def load(self, path=False):
        pass

    # ???
    def extract(self, path=False):
        pass

    def attach_diffs(self, var, new_var):
        prev = self.jpg_data[0][var]
        for row in self.jpg_data:
            curr = row[var]
            row[new_var] = curr - prev
            prev = curr


    def update_counts(self):
        for i, row in enumerate(self.jpg_data):
            count = row["count"]
            if row["med_diff"] > self.plot_params["trans_thresh"]:
                count = -1
            count *= 1 + (
                self.plot_params['night_mult']
                / (1 + 150**(row['from_midnight']-4.5))
            )

            row["new_count"] = count

        for i, shift in self.user_edits.items():
            self.jpg_data[i]["new_count"] = shift


    def update_events(self):
        prev = self.jpg_data[0]
        for i, curr in enumerate(self.jpg_data):
            prev['selected'] = (
                prev['new_count'] > self.plot_params["resp_thresh"]
                or curr['new_count'] > self.plot_params["resp_thresh"]
            )

            if i == self.length-1:
                curr['selected'] = (
                    curr['new_count'] > self.plot_params["resp_thresh"]
                )
            prev = curr

        for move in (1,-1):
            prev = self.jpg_data[-(move<0)]
            for i in range(0, self.length, move):
                curr = self.jpg_data[i]
                if (not curr['selected'] and prev['selected']
                    and curr['new_count'] >= 0):
                    if move == 1:
                        curr['selected'] = (
                            curr['td_minutes']
                            <= self.plot_params["smooth_time"]
                        )
                    elif move == -1:
                        curr['selected'] = (
                            prev['td_minutes']
                            <= self.plot_params["smooth_time"]
                        )
                prev = curr


    def plot(self):

        SLIDER_PARAMS = [
            ('RESP', 0.08, 40000, self.plot_params['resp_thresh'], '%i'),
            ('TRANS', 0.06, 60, self.plot_params['trans_thresh'], '%1.1f'),
            ('SMOOTH', 0.04, 10, self.plot_params['smooth_time'], '%1.1f'),
            ('NIGHT', 0.02, 50, self.plot_params['night_mult'], '%i')
        ]

        def update():
            self.update_counts()
            self.update_events()
            draw()

        def draw():
            for name in self.lines.keys():
                self.lines[name].remove()

            np_counts = np.array(extract_var(self.jpg_data, 'new_count'))
            
            self.lines['edited'] = ax.fill_between(
                np.arange(0, self.length) + 0.5, 0, 1e5,
                where=np_counts<0,
                facecolor='#D8BFAA', alpha=0.5
            )
            self.lines['selected'] = ax.fill_between(
                np.arange(0, self.length) + 0.5, 0, 1e5,
                where=extract_var(self.jpg_data, 'selected'),
                facecolor='#F00314'
            )
            self.lines['count'] = ax.fill_between(
                range(self.length), 0, np_counts,
                facecolor = 'black'
            )
            self.lines['threshold'] = ax.axhline(
                self.plot_params['resp_thresh'], color='#14B37D'
            )

            fig.canvas.draw_idle()

        def on_slide(val):
            for key in self.plot_params.keys():
                slider_key = key[:key.find('_')].upper()
                self.plot_params[key] = self.sliders[slider_key].val
            update()

        def image_pano(xdata):
            i = int(round(xdata))
            array = np.array(range(i - 2, i + 2))
            while True:
                if any(n < 0 for n in array):
                    array += 1
                elif any(n >= self.length for n in array):
                    array -= 1
                else:
                    break

            stack = []
            for n in array:
                if n in self.buffer[0]:
                    ind = self.buffer[0].index(n)
                    img = self.buffer[1][ind]
                    self.buffer[0].append(self.buffer[0].pop(ind))
                    self.buffer[1].append(self.buffer[1].pop(ind))
                else:
                    img = cv2.imread(self.jpg_data[n]['filepath'])
                    self.buffer[0] = self.buffer[0][1:] + [n]
                    self.buffer[1] = self.buffer[1][1:] + [img]

                if self.toggle:
                    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                
                stack.append(img)

            pano = np.hstack(stack)
            h, w, *_ = pano.shape

            cv2.namedWindow("Main", cv2.WINDOW_NORMAL)
            cv2.imshow("Main", cv2.resize(pano, (w // 2, h // 2)))

        def on_click(event):
            if event.dblclick and event.xdata is not None:
                image_pano(event.xdata)

        def on_key(event):
            if event.xdata is not None:
                if self.press[0] is None:
                    i = int(round(event.xdata))
                    if event.key == 'z':
                        self.press = [-1, i]
                    elif event.key == 'x':
                        self.press = [1e5, i]
                    elif event.key == ',':
                        self.press = [0, i]
                if event.key in 'zxc,':
                    image_pano(event.xdata)
                elif event.key == '`':
                    self.toggle = not self.toggle
                    image_pano(event.xdata)
                elif event.key == '.':
                    self.user_edits = {}
                    update()

        def off_key(event):
            try:
                if event.xdata is not None:
                    i = int(round(event.xdata))
                    low, high = sorted((self.press[1], i))
                    lo_to_hi = range(max(0, low), min(self.length, high+1))
                    if event.key in 'zx':
                        new_edits = {i: self.press[0] for i in lo_to_hi}
                        self.user_edits = {**self.user_edits, **new_edits}
                    elif event.key == ',':
                        for i in lo_to_hi:
                            self.user_edits.pop(i, None)
                    self.press = [None, None]
                update()
            except:
                pass

        plt.rc('font', **{'size': 8})
        plt.rcParams['keymap.back'] = 'left, backspace'

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.canvas.set_window_title('Response Filtering')
        fig.subplots_adjust(0.07, 0.18, 0.97, 0.97)

        ax.grid(alpha = 0.4)
        ax.set_axisbelow(True)
        ax.set_ylim([0, 40000])
        ax.set_xlim([0, 500])

        ax.set_xlabel("Frame")
        ax.set_ylabel("Response")
        plt.yticks(rotation=45)

        for name in ('count', 'threshold', 'selected', 'edited'):
            self.lines[name] = ax.axhline()

        for name, pos, max_val, init, fmt in SLIDER_PARAMS:
            slider_ax = fig.add_axes([0.125, pos, 0.8, 0.02])
            self.sliders[name] =  Slider(
                slider_ax, name, 0, max_val,
                valinit=init, valfmt = fmt, color='#003459', alpha=0.5
            )
            self.sliders[name].on_changed(on_slide)

        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('key_release_event', off_key)
        fig.canvas.mpl_connect('button_press_event', on_click)

        ax.fill_between(
            np.arange(0, self.length) + 0.5, 0, 1e5,
            where=[x['from_midnight'] < 4.5 for x in self.jpg_data],
            facecolor='#003459', alpha=0.5
        )

        update()
        plt.show()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    home_path = '/Users/user/Data/python/wardle/2AL (35)'

    first = time.time()

    st = time.time()
    jpg_paths = find_jpgs(home_path)
    end = time.time()
    print(f"A\t{end - st}")

    st = time.time()
    jpg_data = sorted(attach_exif(jpg_paths), key=itemgetter('datetime'))
    end = time.time()
    print(f"B\t{end - st}")

    st = time.time()
    params = generate_clone_params((882,979,0,203), "right")
    processed_data = process_jpgs(jpg_data[:100],
                                  crop=(0,100,0,0),
                                  clone_params=params)
    end = time.time()
    print(f"C\t{end - st}")

    st = time.time()
    cam = Cam(processed_data)
    end = time.time()
    print(f"D\t{end - st}")
    print(cam.jpg_data[0])

    st = time.time()
    cam.update_counts()
    end = time.time()
    print(f"E\t{end - st}")
    print(cam.jpg_data[0])

    st = time.time()
    cam.update_events()
    end = time.time()
    print(f"F\t{end - st}")
    print(cam.jpg_data[0])

    cam.plot()

    last = time.time()
    print(f"END\t{last - first}")



'''
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


class Mem():
    def __init__(self):
        pass


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

    def get_deltas(self):
        return [self.datetimes[i] - self.datetimes[i-1] if
                i > 0 else td(0) for
                i in range(len(self.datetimes))]

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
'''
