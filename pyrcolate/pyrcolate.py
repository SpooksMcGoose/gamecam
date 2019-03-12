import datetime
import json
import os
import shutil
import time

import cv2
import exifread as er
import numpy as np

from datetime import datetime as dt, timedelta as td

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    import Tkinter as tk
    from Tkinter import filedialog

# Work-around a weird tkinter / matplotlib interaction.
# Note: I would use TkAgg like below, but causes key bouncing.
root = tk.Tk()
root.withdraw()
root.destroy()

# import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


''' CONSTANTS '''


DIRECTION_MULTS = {
    "A": (-1, -1, 0, 0),
    "B": (1, 1, 0, 0),
    "R": (0, 0, 1, 1),
    "L": (0, 0, -1, -1)
}


''' GENERAL FUNCTIONS '''


def to_24_hour(datetime):
    time = datetime.time()
    return time.hour + time.minute / 60 + time.second / 3600


def extract_var(data, var):
    return [x[var] for x in data]


# Tkinter works strangely inside of a class, so I moved it out here.
def handle_tkinter(mode, init=None):
    root = tk.Tk()
    root.withdraw()

    if init is None:
        init = os.path.expanduser("~")

    if mode == 'savefile':
        output = filedialog.asksaveasfilename(
            initialdir=init,
            filetypes=(("Save files", "*.sav"), ("All files", "*.*"))
        )
    elif mode == 'openfile':
        output = filedialog.askopenfilename(
            initialdir=init,
            title="Select file",
            filetypes=(("Save files", "*.sav"), ("All files", "*.*"))
        )
    elif mode == 'opendir':
        output = filedialog.askdirectory(
            initialdir=init,
            title="Select folder",
        )

    root.update()
    root.destroy()

    return output


def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    for k, v in d.items():
        if k != 'days':
            d[k] = str(d[k]).rjust(2, '0')
    return fmt.format(**d)


''' SPECIFIC FUNCTIONS (ALL BELOW) '''


# Recursive walk finding all JPG images below home_path.
def find_jpgs(dirpath=None):
    if dirpath is None:
        dirpath = handle_tkinter('opendir')

    if dirpath:
        output = []

        for dir_, subdir, files in os.walk(dirpath):
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

    for deep_row in jpg_data:
        row = deep_row.copy()
        file = open(row['filepath'], "rb")
        tags = er.process_file(file, details=False)

        for t, var, anon in tag_format:
            row[var] = anon(tags[t])

        output.append(row)

    return output


# Quickly finds the median tone of a grayscale image.
def hist_median(image, px_count):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    tally = 0
    threshold = px_count / 2
    for i, count in enumerate(hist):
        tally += count
        if tally > threshold:
            return i


# Outputs a pair of tuples, ((to), (from)).
# Useful for cloning out timestamps, aids in preprocessing.
def generate_clone_tuples(clone_to, clone_from):
    clone_to = np.array(clone_to)
    mults = np.array(DIRECTION_MULTS[clone_from[0].upper()])

    a, b, c, d = clone_to
    h, w = b - a, d - c
    h_or_w = np.array([h, h, w, w])

    clone_from = clone_to + (h_or_w*mults)
    return (tuple(clone_to), tuple(clone_from))


''' IMAGE PREPROCESSING '''


def CROPPER(image, CROP):
    y1, y2, x1, x2 = CROP
    return image[y1:y2, x1:x2]


# Crop image, equalize histogram.
def CROP_EQUALIZE(image, CROP, CLONE_TUPS=None):
    image = CROPPER(image, CROP)
    return cv2.equalizeHist(image)


# Crops, clone out timestamp, equalize.
def CROP_CLONE_EQUALIZE(image, CROP, CLONE_TUPS):
    image = CROPPER(image, CROP)

    (a, b, c, d), (e, f, g, h) = CLONE_TUPS
    image[a:b, c:d] = image[e:f, g:h]

    return cv2.equalizeHist(image)


''' FRAME DIFFERENCING METHODS '''


# Bare-bones. Take difference, threshold, and sum the mask.
def SIMPLE(curr, prev, THRESH, KSIZE=None, MIN_AREA=None):
    difference = cv2.absdiff(curr, prev)
    _, mask = cv2.threshold(
        difference,
        THRESH, 255,
        cv2.THRESH_BINARY
    )
    return cv2.countNonZero(mask)


# Difference, blur (amount changes with ksize), mask and sum.
def BLURRED(curr, prev, THRESH, KSIZE=11, MIN_AREA=None):
    difference = cv2.absdiff(curr, prev)
    blurred = cv2.medianBlur(difference, KSIZE)
    _, mask = cv2.threshold(
        blurred,
        THRESH, 255,
        cv2.THRESH_BINARY
    )
    return cv2.countNonZero(mask)


# Like BLURRED, but only sums drawn contours over given limit.
def CONTOURS(curr, prev, THRESH, KSIZE=11, MIN_AREA=100):
    difference = cv2.absdiff(curr, prev)
    blurred = cv2.medianBlur(difference, KSIZE)

    _, mask = cv2.threshold(
        blurred,
        THRESH, 255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )[-2:]

    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            count += area

    return count


''' JPG PROCESSING '''


# Requires data with filepaths.
# Images may be cropped and cloned, depending on PREPROCESS function.
# Returns a count value after image pair is sent to METHOD function.
# If no CROP specified, CROP will equal dimensions of first photo.
def process_jpgs(
    jpg_data,
    METHOD=CONTOURS,
    CROP=False, CLONE_TUPS=False,
    THRESH=False, KSIZE=11, MIN_AREA=100
):

    if not THRESH:
        THRESH_INIT = False
    if not CROP:
        CROP_INIT = False

    if not CLONE_TUPS:
        PREPROCESS = CROP_EQUALIZE
    else:
        PREPROCESS = CROP_CLONE_EQUALIZE

    output = []

    timer = (len(jpg_data) // 10, time.time())
    for i, deep_row in enumerate(jpg_data):
        row = deep_row.copy()
        if i == 0:
            jpg = cv2.imread(row['filepath'], 0)
            h, w = jpg.shape
            if not CROP:
                CROP = (0, h, 0, w)
            PX_COUNT = h*w
            prev = PREPROCESS(jpg, CROP, CLONE_TUPS)
        elif i % timer[0] == 0:
            progress = i * 10 // timer[0]
            elapsed = time.time() - timer[1]
            total = elapsed / (progress / 100)
            remain = strfdelta(
                td(seconds=total - elapsed),
                "{days}:{hours}:{minutes}:{seconds}"
            )
            print(
                f"{progress}% done. "
                f"{remain} left."
            )

        jpg = cv2.imread(row['filepath'], 0)

        row['median'] = hist_median(jpg, PX_COUNT)

        curr = PREPROCESS(jpg, CROP, CLONE_TUPS)

        if not THRESH_INIT:
            THRESH = row['median']*1.05

        try:
            row['count'] = METHOD(curr, prev, THRESH, KSIZE, MIN_AREA)
        except cv2.error as inst:
            if "(-209:Sizes" in str(inst):
                (a, b), (c, d) = curr.shape[:2], prev.shape[:2]
                h, w = min(a, c), min(b, d)
                tup = (0, h, 0, w)
                print(
                    "FUNCTION ABORTED!\n"
                    "Not all images are of same size, "
                    "consider using the CROP parameter.\n"
                    f"Try CROP={tup}."
                )
                return tup
            else:
                print(inst)

        prev = curr
        output.append(row)

    return output


def force_process_jpgs(
    jpg_data,
    METHOD=CONTOURS,
    CROP=False, CLONE_TUPS=False,
    THRESH=False, KSIZE=11, MIN_AREA=100
):
    output = CROP
    while type(output) is tuple or output is False:
        output = process_jpgs(
            jpg_data, METHOD, output, CLONE_TUPS, THRESH, KSIZE, MIN_AREA
        )
    return output


''' !!! THE MEAT AND POTATOES !!! '''


# Requires data from process_jpg(). Object essentially holds data
# used in exporting, parameters for graphing, and plot function.
class Cam():
    def __init__(self, jpg_data=False):
        self.plot_params = {
            "ceiling": 40000,
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

        self.recent_folder = os.path.expanduser("~")

        # Cam objects don't need to have data to initialize.
        # This way, someone can load() older, processed data.
        if jpg_data:
            self.jpg_data = list(jpg_data)
            self.length = len(self.jpg_data)

            med_count = np.median(
                extract_var(self.jpg_data, 'count')
            )
            self.plot_params['ceiling'] = 40000+(med_count - 500)*4
            self.plot_params['resp_thresh'] = self.plot_params['ceiling'] / 2

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
                row['selected'] = False
                row['edited'] = False

    # Helps Tkinter initialize at last used path.
    def update_recent_folder(self, path):
        if os.path.isfile(path):
            self.recent_folder = os.path.dirname(path)
        else:
            self.recent_folder = path

    # Dump a JSON object with jpg_data, plot_params, and user_edits.
    def save(self, path=False):
        if path:
            filename = path
        else:
            filename = handle_tkinter('savefile', self.recent_folder)
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
            self.update_recent_folder(filename)
        else:
            print("NO SAVE FILEPATH SPECIFIED!")

    # Loads a .sav file, the plot will be identical to what is saved.
    def load(self, path=False):
        if path:
            filename = path
        else:
            filename = handle_tkinter('openfile', self.recent_folder)
        self.update_recent_folder(filename)
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
            print("NO LOAD FILEPATH SPECIFIED!")

    # Any images selected in the plot are exported to a given folder.
    # JPG_data associated with selected images are written to .dat file.
    def export(self, path=False):
        if path:
            directory = path
        else:
            directory = handle_tkinter('opendir', self.recent_folder)
        self.update_recent_folder(directory)
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
            if write_data:
                with open(os.path.join(directory, '_export.dat'), 'w') as f:
                    variables = sorted(write_data[0].keys())
                    for i, row in enumerate(write_data):
                        if i != 0:
                            f.write('\n')
                        else:
                            f.write('\t'.join(variables) + '\n')
                        f.write('\t'.join(str(row[v]) for v in variables))
            else:
                print("NO IMAGES SELECTED FOR EXPORT!")
        else:
            print("NO EXPORT DIRECTORY SPECIFIED!")

    # Finds the difference between the current and previous variable.
    def attach_diffs(self, var, new_var):
        prev = self.jpg_data[0][var]
        for row in self.jpg_data:
            curr = row[var]
            row[new_var] = curr - prev
            prev = curr

    # Marks which photos to label as edited based on index.
    def mark_edits(self, i):
        if i == 0:
            self.jpg_data[i]["edited"] = True
        else:
            self.jpg_data[i-1]["edited"] = True
            self.jpg_data[i]["edited"] = True

    # A new_count is generated, modifying the count variable.
    # Day to night transitions are filtered (slider val).
    # Counts associated with night are multiplied (slider val).
    # User edits are applied.
    def update_counts(self):
        for i, row in enumerate(self.jpg_data):
            row['edited'] = False
            new_count = row["count"]
            if row["med_diff"] > self.plot_params["trans_thresh"]:
                new_count = 0
                self.mark_edits(i)
            new_count *= 1 + (
                self.plot_params['night_mult']
                / (1 + 150**(row['from_midnight']-4.5))
            )

            row["new_count"] = new_count

        for i, shift in self.user_edits.items():
            self.jpg_data[i]["new_count"] = shift
            self.mark_edits(i)

    # An event is a contiguous sequence of images.
    # First images are identified as being selected or not.
    # Then, events are lumped based on time since last image (slider val).
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

        for move in (1, -1):
            prev = self.jpg_data[-(move < 0)]
            for i in range(0, self.length, move):
                curr = self.jpg_data[i]
                boo = (
                    not curr['selected']
                    and prev['selected']
                    and curr['new_count'] >= 0
                )
                if boo:
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

    # Interactive plot for selected desired images.
    def plot(self):

        CEIL_X = self.plot_params['ceiling']*1.5
        SLIDER_PARAMS = [
            ('RESP', 0.08, CEIL_X, self.plot_params['resp_thresh'], '%i'),
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

            CEIL_X = self.plot_params['ceiling']*1.5
            np_counts = np.array(extract_var(self.jpg_data, 'new_count'))

            self.lines['edited'] = ax.fill_between(
                np.arange(0, self.length) + 0.5, 0, CEIL_X,
                where=extract_var(self.jpg_data, 'edited'),
                facecolor='#D8BFAA', alpha=0.5
            )
            self.lines['selected'] = ax.fill_between(
                np.arange(0, self.length) + 0.5, 0, CEIL_X,
                where=extract_var(self.jpg_data, 'selected'),
                facecolor='#F00314'
            )
            self.lines['count'] = ax.fill_between(
                range(self.length), 0, np_counts,
                facecolor='black'
            )
            self.lines['threshold'] = ax.axhline(
                self.plot_params['resp_thresh'], color='#14B37D'
            )

            fig.canvas.draw_idle()

        def on_slide(val):
            for key in self.plot_params.keys():
                if key != 'ceiling':
                    slider_key = key[:key.find('_')].upper()
                    self.plot_params[key] = self.sliders[slider_key].val
            update()

        # Displays last two and next two images from response spike.
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
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.cv2.equalizeHist(gray)

                stack.append(img)

            min_y = min(img.shape[0] for img in stack)

            pano = np.hstack((img[:min_y, :] for img in stack))
            h, w, *_ = pano.shape

            cv2.namedWindow("Gallery", cv2.WINDOW_NORMAL)
            cv2.imshow("Gallery", cv2.resize(pano, (w // 2, h // 2)))

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
                elif event.key == 'v':
                    self.toggle = not self.toggle
                    image_pano(event.xdata)
                elif event.key == '.':
                    self.user_edits = {}

        def off_key(event):
            try:
                if event.xdata is not None and event.key in 'zx,':
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
            except TypeError:
                pass

        plt.rc('font', **{'size': 8})
        plt.rcParams['keymap.back'] = 'left, backspace'

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.canvas.set_window_title('Response Filtering')
        fig.subplots_adjust(0.07, 0.18, 0.97, 0.97)

        ax.grid(alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_ylim([0, self.plot_params['ceiling']])
        ax.set_xlim([0, min(500, self.length)])

        ax.set_xlabel("Frame")
        ax.set_ylabel("Response")
        plt.yticks(rotation=45)

        for name in ('count', 'threshold', 'selected', 'edited'):
            self.lines[name] = ax.axhline()

        for name, pos, max_val, init, fmt in SLIDER_PARAMS:
            slider_ax = fig.add_axes([0.125, pos, 0.8, 0.02])
            self.sliders[name] = Slider(
                slider_ax, name, 0, max_val,
                valinit=init, valfmt=fmt, color='#003459', alpha=0.5
            )
            self.sliders[name].on_changed(on_slide)

        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('key_release_event', off_key)
        fig.canvas.mpl_connect('button_press_event', on_click)

        ax.fill_between(
            np.arange(0, self.length) + 0.5, 0, CEIL_X,
            where=[x['from_midnight'] < 4.5 for x in self.jpg_data],
            facecolor='#003459', alpha=0.5
        )

        update()
        plt.show()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("1) Please navigate to folder with camera-trapping images.")
    jpg_paths = find_jpgs()

    jpg_data = attach_exif(jpg_paths)
    jpg_data.sort(key=lambda x: x['datetime'])

    print("2) Images are being processed.")
    processed_data = process_jpgs(jpg_data)

    if type(processed_data) is not tuple:
        cam = Cam(processed_data)

        print("3) Please choose a location for an initial save.")
        cam.save()

        print("4) Use the interactive plot to select images for export.")
        print('\n'.join((
            "QUICK GUIDE:",
            "   c - Hold to VIEW IMAGES in gallery.",
            "   x - Hold and release to INCREASE response value to 1e5.",
            "   z - Hold and release to DECREASE response value to -1.",
            "   , - Hold and release to REMOVE EDITS.",
            "   . - Press to RESET ALL EDITS.",
            "   v - Press for EQUALIZED image (for dark images)."
        )))
        cam.plot()

        print("5) Save once again, so changes are recorded.")
        cam.save()

        print("6) Finally, choose a location for export.")
        cam.export()
