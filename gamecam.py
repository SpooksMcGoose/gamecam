import os
import json
import time
import datetime
from datetime import datetime as dt, timedelta as td
from operator import itemgetter

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    import Tkinter as tk
    from Tkinter import filedialog
# Get around a weird tkinter / matplotlib interaction.
# Note: I would use TkAgg like below, but causes key bouncing.
root = tk.Tk()
root.withdraw()

# import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

import exifread as er
import numpy as np

import cv2
import shutil


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


def most_recent(path):
    files = sorted(f for f in os.listdir(path) if f.endswith('.dat'))
    return os.path.join(path, files[-1])


def handle_tkinter(mode):
    root = tk.Tk()
    root.withdraw()

    if mode == 'savefile':
        output = filedialog.asksaveasfilename(
            initialdir = os.path.expanduser("~"),
            title = "Select file",
            filetypes = (("Save files","*.sav"),("All files","*.*"))
        )
    elif mode == 'openfile':
        output = filedialog.askopenfilename(
            initialdir = os.path.expanduser("~"),
            title = "Select file",
            filetypes = (("Save files","*.sav"),("All files","*.*"))
        )
    elif mode == 'opendir':
        output = filedialog.askdirectory(
            initialdir = os.path.expanduser("~"),
            title = "Select folder",
        )

    root.update()
    root.destroy()

    return output


############################################################
###                 SPECIFIC   FUNCTIONS                 ###
############################################################


# Recursive walk finding all JPG images below home_path.
def find_jpgs(home_path):
    if not home_path:
        home_path = handle_tkinter('opendir')

    if home_path:
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
    else:
        print("Please provide an image directory.")


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

        contours, _ = cv2.findContours(
            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )[-2:]

        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                count += area
        row['count'] = count

        prev = curr
        output.append(row)

    return output


# Minimum input requires process_jpg data. Object essentially holds
# data used in exporting, parameters for graphing, and plot function.
class Cam():
    def __init__(self, jpg_data=False):
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

    def save(self):
        filename = handle_tkinter('savefile')
        if filename:
            temp_data = [
                {k: v for k, v in row.items()}
                for row in self.jpg_data
            ]
            for row in temp_data:
                if 'datetime' in row.keys():
                    row['datetime'] = dt.strftime(
                        row['datetime'], "%Y-%m-%d %H:%M:%S"
                    )
                if 'timedelta' in row.keys():
                    row['timedelta'] = row['timedelta'].total_seconds()
                if 'selected' in row.keys():
                    row['selected'] = int(row['selected'])

            with open(filename, 'w') as f:
                f.write(json.dumps(self.plot_params) + '\n')
                f.write(json.dumps(temp_data) + '\n')
                f.write(json.dumps(self.user_edits))
        else:
            print("Please provide a filename to save.")

    def load(self):
        filename = handle_tkinter('openfile')
        if filename:
            with open(filename, 'r') as f:
                self.plot_params = json.loads(next(f))
                temp_data = json.loads(next(f))
                temp_dict = json.loads(next(f))

            for row in temp_data:
                if 'datetime' in row.keys():
                    row['datetime'] = dt.strptime(
                        row['datetime'], "%Y-%m-%d %H:%M:%S"
                    )
                if 'timedelta' in row.keys():
                    row['timedelta'] = td(seconds=row['timedelta'])
                if 'selected' in row.keys():
                    row['selected'] = bool(row['selected'])
            self.jpg_data = temp_data.copy()
            self.length = len(self.jpg_data)

            self.user_edits = {int(k): v for k, v in temp_dict.items()}
        else:
            print("Please provide a filepath to load.")

    def export(self):
        directory = handle_tkinter('opendir')
        if directory:
            write_data = []

            for row in self.jpg_data:
                if row['selected']:
                    write_data.append(row.copy())
                    dt_ISO = dt.strftime(row['datetime'], "%Y%m%dT%H%M%S")
                    new_path = os.path.join(
                        directory, '_'.join((dt_ISO, row['filename']))
                    )
                    shutil.copy2(row['filepath'], new_path)

            with open(os.path.join(directory, '_export.dat'), 'w') as f:
                variables = sorted(write_data[0].keys())
                for i, row in enumerate(write_data):
                    if i != 0:
                        f.write('\n')
                    else:
                        f.write('\t'.join(variables) + '\n')
                    f.write('\t'.join(str(row[v]) for v in variables))
        else:
            print("Please provide an export directory.")

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
            ('RESP', 0.08, 60000, self.plot_params['resp_thresh'], '%i'),
            ('TRANS', 0.06, 120, self.plot_params['trans_thresh'], '%1.1f'),
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
    home_path = '/Users/user/Data/python/wardle/2AL (35)'

    st = time.time()
    jpg_paths = find_jpgs(home_path)
    end = time.time()
    print(f"A\t{end - st}")
    #print(jpg_paths[0])

    st = time.time()
    jpg_data = sorted(attach_exif(jpg_paths), key=itemgetter('datetime'))
    end = time.time()
    print(f"B\t{end - st}")
    #print(jpg_data[0])

    st = time.time()
    params = generate_clone_params((882,979,0,203), "right")
    processed_data = process_jpgs(jpg_data[:20],
                                  crop=(0,100,0,0),
                                  clone_params=params)
    end = time.time()
    print(f"C\t{end - st}")
    #print(processed_data[0])

    st = time.time()
    cam = Cam(processed_data)
    end = time.time()
    print(f"D\t{end - st}")
    #print(cam.jpg_data[0])

    st = time.time()
    cam.update_counts()
    end = time.time()
    print(f"E\t{end - st}")
    #print(cam.jpg_data[0])

    cam.plot()
    cam.save()

    wam = Cam()
    wam.load()
    wam.plot()
    wam.export()
