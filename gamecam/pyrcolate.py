import json
import operator
import os
import shutil
import sys
import time

import cv2
import exifread as er
import numpy as np

from gamecam import handle_tkinter, find_imgs
from datetime import datetime as dt, timedelta as td

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


# CONSTANTS


GUIDE = "\n".join((
    "QUICK GUIDE:",
    "   c - Hold to VIEW IMAGES in gallery.",
    "   x - Hold and release to INCREASE response value to 1e5.",
    "   z - Hold and release to DECREASE response value to -1.",
    "   , - Hold and release to REMOVE EDITS.",
    "   . - Press to RESET ALL EDITS.",
    "   v - Press for EQUALIZED image (for dark images)."
))


def PARSE_DT(raw):
    """Default parser for EXIF "Image DateTime" tag.
    """

    return dt.strptime(str(raw), "%Y:%m:%d %H:%M:%S")


def SORT_BY_DT(row):
    """Default sort for construct_jpg_data().
    """

    return row["datetime"]


DEFAULT_PARSE = [
    ("Image DateTime", "datetime", PARSE_DT)
]

DEFAULT_PLOT_PARAMS = {
    "ceiling": 40000,
    "resp_thresh": 20000,
    "trans_thresh": 30,
    "smooth_time": 1,
    "night_mult": 0
}

DIRECTION_MULTS = {
    "A": (-1, -1, 0, 0),
    "B": (1, 1, 0, 0),
    "R": (0, 0, 1, 1),
    "L": (0, 0, -1, -1)
}


# GENERAL FUNCTIONS


def to_24_hour(datetime):
    """Converts datetime.datetime type to 24 hour float type.
    """

    time = datetime.time()
    return time.hour + time.minute / 60 + time.second / 3600


def extract_var(data, var):
    """Returns a list of values corresponding to a variable name.

    >>> foo = [{"name": "Shane", "age": 22}, {"name": "Eve", "age": 7}]
    >>> extract_var(data=foo, var="name")
    ["Shane", "Eve"]
    """

    return [x[var] for x in data]


def strfdelta(tdelta, fmt):
    """Formats a timedelta object as a string.

    Parameters
    ----------
    tdelta : datetime.timedelta
        Timedelta object to format.
    fmt : str
        Contains format calls to days, hours, minutes, and seconds.

    Returns
    -------
    str
        Right justified 2 spaces, filled with zeros.
    """

    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    for k, v in d.items():
        if k != "days":
            d[k] = str(d[k]).rjust(2, "0")
    return fmt.format(**d)


# SPECIFIC FUNCTIONS (ALL BELOW)


def attach_exif(jpg_data, parse_tags=DEFAULT_PARSE):
    """Loops through jpg_data, reading filepaths and attaching EXIF data.

    Parameters
    ----------
    jpg_data : list of dictionaries
        Requires that "filepath" is in dictionary keys, which is easily
        provided by find_imgs() prior to this function.
    parse_tags : list of tuples, optional
        By default, only Image DateTime is retrieved from EXIF data using
        DEFAULT_PARSE. Examine DEFAULT_PARSE as an example parameter to
        pass to attach_exif(), if more data is desired from EXIF tags.

    Returns
    -------
    list of dictionaries
        Same as jpg_data, but now with desired EXIF data attached.
    """

    output = []

    for deep_row in jpg_data:
        row = deep_row.copy()
        file = open(row["filepath"], "rb")
        tags = er.process_file(file, details=False)

        for t, var, anon in parse_tags:
            row[var] = anon(tags[t])

        output.append(row)

    return output


def hist_median(image):
    """Quickly finds the median tone of a grayscale image.
    """

    px_count = image.shape[0] * image.shape[1]
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    tally = 0
    threshold = px_count / 2
    for i, count in enumerate(hist):
        tally += count
        if tally > threshold:
            return i


def generate_clone_tuples(clone_to, fill_from):
    """Loops through jpg_data, reading filepaths and attaching EXIF data.

    Useful for cloning out timestamps, aids in histogram equalization.

    Parameters
    ----------
    clone_to : tuple
        Format is (y1, y2, x1, x2), just like slicing images as np.arrays.
    fill_from : {"A", "B", "R", "L"}
        Calls directional tuples from DIRECTION_MULTS to fill pixels within
        the clone_to area. Neighboring pixels from "[A]bove," "[B]elow,"
        "[R]ight," and "[L]eft" are used for filling the area.

    Returns
    -------
    pair of tuples
        Matches the format for the clone parameter in process_jpgs().
    """

    clone_to = np.array(clone_to)
    mults = np.array(DIRECTION_MULTS[clone_from[0].upper()])

    a, b, c, d = clone_to
    h, w = b - a, d - c
    h_or_w = np.array([h, h, w, w])

    clone_from = clone_to + (h_or_w*mults)
    return (tuple(clone_to), tuple(clone_from))


# IMAGE PREPROCESSING


def CROPPER(image, crop):
    """Returns a cropped image.

    Parameters
    ----------
    image : numpy.ndarray
        Image array, typically from cv2.imread().
    crop : tuple
        Format is (y1, y2, x1, x2), just like slicing images as np.arrays.
    """

    y1, y2, x1, x2 = crop
    return image[y1:y2, x1:x2]


def CROP_EQUALIZE(image, crop, clone=None):
    """Returns a cropped image with an equalized histogram.

    Parameters
    ----------
    image : numpy.ndarray
        Image array, typically from cv2.imread().
    crop : tuple
        Format is (y1, y2, x1, x2), just like slicing images as np.arrays.
    clone : pair of tuples, optional
        Matches the format ((clone_to), (clone_from)). For simplicity,
        use generate_clone_tuples() to generate this object.
    """

    image = CROPPER(image, crop)
    return cv2.equalizeHist(image)


def CROP_CLONE_EQUALIZE(image, crop, clone):
    """Returns a cropped image, with specified cloning and equalization.

    Parameters
    ----------
    image : numpy.ndarray
        Image array, typically from cv2.imread().
    crop : tuple
        Format is (y1, y2, x1, x2), just like slicing images as np.arrays.
    clone : pair of tuples
        Matches the format ((clone_to), (clone_from)). For simplicity,
        use generate_clone_tuples() to generate this object.
    """

    image = CROPPER(image, crop)

    (a, b, c, d), (e, f, g, h) = clone
    image[a:b, c:d] = image[e:f, g:h]

    return cv2.equalizeHist(image)


# FRAME DIFFERENCING METHODS


# Bare-bones. Take difference, threshold, and sum the mask.
def SIMPLE(curr, prev, threshold, ksize=None, min_area=None):
    """Most basic frame differencing method.

    Takes two images, then finds their absolute difference. A simple
    threshold is called, the resulting white pixels are counted toward
    response (movement). Very noisy, but fast.

    Parameters
    ----------
    curr : numpy.ndarray
        Image array, typically from cv2.imread(). One of the two images
        for the absolute difference to be taken.
    prev : numpy.ndarray
        Like curr. The second image to be differenced.
    threshold : int, in range(0, 256)
        Parameter to be passed to the cv2.threshold() function.
    ksize : int, unused
        Used in the BLURRED() and CONTOURS() functions, but retained here to
        shorten the process_jpgs() function.
    min_area : int, unused
        Only used in the CONTOURS() function, but retained here to shorten
        the process_jpgs() function.
    """

    difference = cv2.absdiff(curr, prev)
    _, mask = cv2.threshold(
        difference,
        threshold, 255,
        cv2.THRESH_BINARY
    )
    return cv2.countNonZero(mask)


# Difference, blur (amount changes with ksize), mask and sum.
def BLURRED(curr, prev, threshold, ksize=11, min_area=None):
    """Useful, mid-grade frame differencing method.

    Takes two images, then finds their absolute difference. Prior to
    thresholding, the differenced image is blurred to reduce noise.
    After thresholding, the resulting white pixels are counted toward
    response (movement). Works decently, a little faster than COUNTOURS.

    Parameters
    ----------
    curr : numpy.ndarray
        Image array, typically from cv2.imread(). One of the two images
        for the absolute difference to be taken.
    prev : numpy.ndarray
        Like curr. The second image to be differenced.
    threshold : int, in range(0, 256)
        Parameter to be passed to the cv2.threshold() function.
    ksize : int
        Parameter to be passed to the cv2.medianBlur() function.
        Default is 11. Must be positive, odd number.
    min_area : int, unused
        Only used in the CONTOURS() function, but retained here to shorten
        the process_jpgs() function.
    """

    difference = cv2.absdiff(curr, prev)
    blurred = cv2.medianBlur(difference, ksize)
    _, mask = cv2.threshold(
        blurred,
        threshold, 255,
        cv2.THRESH_BINARY
    )
    return cv2.countNonZero(mask)


# Like BLURRED, but only sums drawn contours over given limit.
def CONTOURS(curr, prev, threshold, ksize=11, min_area=100):
    """Slower, but powerful frame differencing method.

    Takes two images, then finds their absolute difference. Prior to
    thresholding, the differenced image is blurred to reduce noise.
    After thresholding, contours are drawn around the resulting white pixels.
    If the contours are above the min_area parameter, they are counted as a
    response (movement). Works very well, little noise; slower than others.

    Parameters
    ----------
    curr : numpy.ndarray
        Image array, typically from cv2.imread(). One of the two images
        for the absolute difference to be taken.
    prev : numpy.ndarray
        Like curr. The second image to be differenced.
    threshold : int, in range(0, 256)
        Parameter to be passed to the cv2.threshold() function.
    ksize : int
        Parameter to be passed to the cv2.medianBlur() function.
        Default is 11. Must be positive, odd number.
    min_area : int
        Minimum contour area to count as a response (movement). Default is
        an area of 100 pixels. Larger numbers decreases sensitivity.
    """

    difference = cv2.absdiff(curr, prev)
    blurred = cv2.medianBlur(difference, ksize)

    _, mask = cv2.threshold(
        blurred,
        threshold, 255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )[-2:]

    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            count += area

    return count


# JPG PROCESSING


def process_jpgs(
    jpg_data,
    method=CONTOURS,
    crop=False, clone=False,
    threshold=False, ksize=11, min_area=100
):
    """Generates a response (movement) metric between images.

    Works hierarchically to preform image cropping, cloning, and histogram
    equalization on images read from jpg_data filepaths before being passed
    on to frame differencing methods that generate a response metric.

    This is the last step before the jpg_data list can be fed into the
    Cam() class for response filtering.

    Parameters
    ----------
    jpg_data : list of dictionaries
        Requires that "filepath" is in dictionary keys, which is easily
        provided by find_imgs() prior to this function.
    method : function, {SIMPLE, BLURRED, COUNTOURS}
        Determines the frame differencing method to use. Ordered from
        left to right based on increasing accuracy, decreasing speed.
    crop : tuple
        Format is (y1, y2, x1, x2), just like slicing images as np.arrays.
    clone : pair of tuples
        Matches the format ((clone_to), (clone_from)). For simplicity,
        use generate_clone_tuples() to generate this object.
    threshold : int, in range(0, 256)
        Parameter to be passed to the cv2.threshold() function.
    ksize : int
        Parameter to be passed to the cv2.medianBlur() function.
        Default is 11. Must be positive, odd number.
    min_area : int
        Minimum contour area to count as a response (movement). Default is
        an area of 100 pixels. Larger numbers decreases sensitivity.

    Returns
    -------
    list of dictionaries
        Same as incoming jpg_data, but now with the median image tone and
        a count variable, which respresents how many pixels have changed
        between a photo and its previous, after preprocessing / thresholding.
    """

    if not threshold:
        thresh_init = False

    if not clone:
        preprocess = CROP_EQUALIZE
    else:
        preprocess = CROP_CLONE_EQUALIZE

    output = []

    timer = (len(jpg_data) // 10, time.time())
    for i, deep_row in enumerate(jpg_data):
        row = deep_row.copy()
        if i == 0:
            jpg = cv2.imread(row["filepath"], 0)
            h, w = jpg.shape
            if not crop:
                crop = (0, h, 0, w)
            prev = preprocess(jpg, crop, clone)
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

        jpg = cv2.imread(row["filepath"], 0)

        row["median"] = hist_median(jpg)

        curr = preprocess(jpg, crop, clone)

        if not thresh_init:
            threshold = row["median"]*1.05

        try:
            row["count"] = method(curr, prev, threshold, ksize, min_area)
        except cv2.error as inst:
            if "(-209:Sizes" in str(inst):
                (a, b), (c, d) = curr.shape[:2], prev.shape[:2]
                h, w = min(a, c), min(b, d)
                tup = (0, h, 0, w)
                print(
                    "FUNCTION ABORTED!\n"
                    "Not all images are of same size, "
                    "consider using the crop parameter.\n"
                    f"Try crop={tup}."
                )
                return tup
            else:
                print(inst)

        prev = curr
        output.append(row)

    return output


def construct_jpg_data(
    dirpath=None,
    parse_tags=DEFAULT_PARSE,
    sort_key=SORT_BY_DT,
    process_options={}
):
    """Performs all necessary steps to make jpg_data feedable to Cam().

    Parameters
    ----------
    dirpath : str, optional
        If no path provided, the user can navigate to it via tkinter window.
    parse_tags : list of tuples, optional
        By default, only Image DateTime is retrieved from EXIF data using
        DEFAULT_PARSE. Examine DEFAULT_PARSE as an example parameter to
        pass to attach_exif(), if more data is desired from EXIF tags.
    sort_key : function, optional
        By default, dictionaries within the jpg_data list are sorted by
        their "Image DateTime" EXIF tag. This can be changed if images don"t
        have a datetime, but do have a variable for sequence.
    process_options : dictionary, optional
        Passes along paramters to the process_jpgs() function. By default,
        no options are specified. Make sure that this parameter is mappable.

    Returns
    -------
    list of dictionaries
        Each row contains everything that"s needed to feed a Cam() object
        with its initial data.
    """

    if dirpath is None:
        print("Navigate to image folder.")
    jpg_paths = find_imgs(dirpath)
    print(f"{len(jpg_paths)} images found.")
    jpg_data = attach_exif(jpg_paths, parse_tags)
    jpg_data.sort(key=sort_key)
    print("Started processing...")
    output = process_jpgs(jpg_data, **process_options)
    print("Done!")
    return output


# THE MEAT AND POTATOES


# Requires data from process_jpg(). Object essentially holds data
# used in exporting, parameters for graphing, and plot function.
class Cam():
    """
    A class used to store, plot, filter, and export image data.

    Contains the heart of this module, the interactive plot() method.
    A simply initialization could be: Cam(construct_jpg_data()). After
    filtering images with the plot, save() and export() should be called
    so as not to lose any prior work.

    Attributes
    ----------
    jpg_data : list of dictionaries
        Similar to what is inputted, but includes variables denoting
        selection, edits, event, etc.
    plot_params : dictionary
        Parameters used in the plot() method. To reset these values, re-
        assign DEFAULT_PLOT_PARAMS to this attribute.

    Methods
    -------
    attach_diffs(var, new_var)
        Finds the difference between the current and previous items.
    export(path=False)
        Exports selected images and a .csv file.
    load(path=False)
        Loads a .sav file.
    mark_edits(i)
        Marks which photos have been edited.
    plot()
        Interactive plot used to select images for export.
    save(path=False)
        Dumps a JSON object as a .sav file.
    update_counts()
        Updates jpg_data attribute with new counts.
    update_events()
        Updates jpg_data attribute with new events.
    update_recent_folder(path)
        Provides Tkinter calls with the last used path.
    """

    def __init__(self, jpg_data=False, resp_var="count"):
        """
        Parameters
        ----------
        jpg_data : list of dictionaries, optional
            Typically requires the output of process_jpgs(). Can be omitted
            if a empty Cam() object is desired for a load() method call.
        resp_var : str, optional
            A key found in jpg_data, typically the "count" variable is used
            as a response, but alternatives like "median" can be used to plot
            jpgs without processing them first.
        """

        self.resp_var = resp_var

        self.plot_params = DEFAULT_PLOT_PARAMS

        self.lines = {}
        self.sliders = {}

        self.user_edits = {}
        self.press = [None, None]
        self.toggle = False

        self.buffer = [[None] * 64, [None] * 64]

        self.recent_folder = os.path.expanduser("~")

        # Cam objects don"t need to have data to initialize.
        # This way, someone can load() older, processed data.
        if jpg_data:
            self.jpg_data = list(jpg_data)
            self.length = len(self.jpg_data)

            self.dt_present = "datetime" in self.jpg_data[0].keys()

            for row in self.jpg_data:
                row["count"] = row[self.resp_var]

            self.plot_params["ceiling"] = np.percentile(
                extract_var(self.jpg_data, "count"), 80
            )
            self.plot_params["resp_thresh"] = self.plot_params["ceiling"] / 2

            self.attach_diffs("median", "med_diff")

            for row in self.jpg_data:
                row["med_diff"] = abs(row["med_diff"])
                row["selected"] = False
                row["edited"] = False

            if self.dt_present:
                self.attach_diffs("datetime", "timedelta")
                for row in self.jpg_data:
                    hour = to_24_hour(row["datetime"])
                    row["from_midnight"] = (
                        hour if hour < 12
                        else (hour - 24) * -1
                    )
                    row["td_minutes"] = round(
                        row["timedelta"].total_seconds() / 60, 2
                    )

    def update_recent_folder(self, path):
        """Provides Tkinter calls with the last used path.
        """

        if os.path.isfile(path):
            self.recent_folder = os.path.dirname(path)
        else:
            self.recent_folder = path

    def save(self, path=False):
        """Dumps a JSON object with jpg_data, plot_params, and user_edits.
        """

        if path:
            filename = path
        else:
            filename = handle_tkinter("savefile", self.recent_folder)
        if filename:
            temp_data = [
                {k: v for k, v in row.items()}
                for row in self.jpg_data
            ]
            for row in temp_data:
                if "datetime" in row.keys():
                    row["datetime"] = dt.strftime(
                        row["datetime"], "%Y-%m-%d %H:%M:%S"
                    )
                if "timedelta" in row.keys():
                    row["timedelta"] = row["timedelta"].total_seconds()
                if "selected" in row.keys():
                    row["selected"] = int(row["selected"])

            with open(filename, "w") as f:
                f.write(json.dumps(self.plot_params) + "\n")
                f.write(json.dumps(temp_data) + "\n")
                f.write(json.dumps(self.user_edits))
            self.update_recent_folder(filename)
        else:
            print("NO SAVE FILEPATH SPECIFIED!")

    def load(self, path=False):
        """Loads a .sav file, the Cam() object is now identical to the last.
        """

        if path:
            filename = path
        else:
            filename = handle_tkinter("openfile", self.recent_folder)
        self.update_recent_folder(filename)
        if filename:
            with open(filename, "r") as f:
                self.plot_params = json.loads(next(f))
                temp_data = json.loads(next(f))
                temp_dict = json.loads(next(f))

            for row in temp_data:
                if "datetime" in row.keys():
                    row["datetime"] = dt.strptime(
                        row["datetime"], "%Y-%m-%d %H:%M:%S"
                    )
                    self.dt_present = True
                else:
                    self.dt_present = False
                if "timedelta" in row.keys():
                    try:
                        row["timedelta"] = td(seconds=row["timedelta"])
                    except AttributeError:
                        pass
                if "selected" in row.keys():
                    row["selected"] = bool(row["selected"])
            self.jpg_data = temp_data.copy()
            self.length = len(self.jpg_data)

            self.user_edits = {int(k): v for k, v in temp_dict.items()}
        else:
            print("NO LOAD FILEPATH SPECIFIED!")

    def export(self, path=False):
        """Exports selected images and a .csv file to specified directory.
        """

        if path:
            directory = path
        else:
            directory = handle_tkinter("opendir", self.recent_folder)
        self.update_recent_folder(directory)
        if directory:
            write_data = []

            for i, row in enumerate(self.jpg_data):
                if row["selected"]:
                    write_data.append(row.copy())
                    if self.dt_present:
                        dt_ISO = dt.strftime(
                            row["datetime"], "%Y%m%dT%H%M%S"
                        )
                    else:
                        dt_ISO = str(i)
                    new_path = os.path.join(
                        directory, "_".join((dt_ISO, row["filename"]))
                    )
                    shutil.copy2(row["filepath"], new_path)
            if write_data:
                with open(os.path.join(directory, "_export.csv"), "w") as f:
                    variables = sorted(write_data[0].keys())
                    for i, row in enumerate(write_data):
                        if i != 0:
                            f.write("\n")
                        else:
                            f.write(",".join(variables) + "\n")
                        f.write(",".join(str(row[v]) for v in variables))
            else:
                print("NO IMAGES SELECTED FOR EXPORT!")
        else:
            print("NO EXPORT DIRECTORY SPECIFIED!")

    def attach_diffs(self, var, new_var):
        """Finds the difference between the current and previous variable.

        Requires a list of dictionaries.
        """

        prev = self.jpg_data[0][var]
        for row in self.jpg_data:
            curr = row[var]
            row[new_var] = curr - prev
            prev = curr

    def mark_edits(self, i):
        """Marks which photos to label as edited based on [i]ndex.
        """

        if i == 0:
            self.jpg_data[i]["edited"] = True
        else:
            self.jpg_data[i-1]["edited"] = True
            self.jpg_data[i]["edited"] = True

    def update_counts(self):
        """Updates jpg_data with new counts (response metric).

        Variable new_count is attached to jpg_data. Day to night transitions
        are filtered out based on slider. Counts taken at with nighttime are
        multiplied based on slider. Manual user edits are applied.
        """

        for i, row in enumerate(self.jpg_data):
            row["edited"] = False
            new_count = row["count"]
            if row["med_diff"] > self.plot_params["trans_thresh"]:
                new_count = 0
                self.mark_edits(i)
            if self.dt_present:
                new_count *= 1 + (
                    self.plot_params["night_mult"]
                    / (1 + 150**(row["from_midnight"]-4.5))
                )

            row["new_count"] = new_count

        for i, shift in self.user_edits.items():
            self.jpg_data[i]["new_count"] = shift
            self.mark_edits(i)

    def update_events(self):
        """Updates jpg_data with new events (runs of detection).

        An event is a contiguous sequence of images. First, images are
        identified as being selected or not. Then, events are lumped based
        on time since last image (SMOOTH slider value).
        """

        prev = self.jpg_data[0]
        for i, curr in enumerate(self.jpg_data):
            prev["selected"] = (
                prev["new_count"] > self.plot_params["resp_thresh"]
                or curr["new_count"] > self.plot_params["resp_thresh"]
            )

            if i == self.length-1:
                curr["selected"] = (
                    curr["new_count"] > self.plot_params["resp_thresh"]
                )
            prev = curr

        if self.dt_present:
            for move in (1, -1):
                prev = self.jpg_data[-(move < 0)]
                for i in range(0, self.length*move, move):
                    curr = self.jpg_data[i]
                    boo = (
                        not curr["selected"]
                        and prev["selected"]
                        and curr["new_count"] >= 0
                    )
                    if boo:
                        if move == 1:
                            curr["selected"] = (
                                curr["td_minutes"]
                                <= self.plot_params["smooth_time"]
                            )
                        elif move == -1:
                            curr["selected"] = (
                                prev["td_minutes"]
                                <= self.plot_params["smooth_time"]
                            )
                    prev = curr
        else:
            nudge = int(self.plot_params["smooth_time"])
            master_set = set()
            for i, row in enumerate(self.jpg_data):
                if row["selected"]:
                    for func in (operator.add, operator.sub):
                        for j in range(nudge+1):
                            ind = func(i, j)
                            try:
                                row = self.jpg_data[ind]
                                if row["new_count"] < 0:
                                    if func == operator.sub:
                                        master_set.add(ind)
                                    break
                                else:
                                    master_set.add(ind)
                            except IndexError:
                                pass
            for i in master_set:
                self.jpg_data[i]["selected"] = True

    def plot(self):
        """Interactive plot used to select images for export.

        QUICK GUIDE:
           c - Hold to VIEW IMAGES in gallery.
           x - Hold and release to INCREASE response value to 1e5.
           z - Hold and release to DECREASE response value to -1.
           , - Hold and release to REMOVE EDITS.
           . - Press to RESET ALL EDITS.
           v - Press for EQUALIZED image (for dark images).
        """

        try:
            self.jpg_data
        except AttributeError as inst:
            raise inst

        CEIL_X = self.plot_params["ceiling"]*1.5
        SLIDER_PARAMS = [
            ("RESP", 0.08, CEIL_X, self.plot_params["resp_thresh"], "%i"),
            ("TRANS", 0.06, 120, self.plot_params["trans_thresh"], "%1.1f"),
            ("SMOOTH", 0.04, 10, self.plot_params["smooth_time"], "%1.1f"),
            ("NIGHT", 0.02, 50, self.plot_params["night_mult"], "%i")
        ]

        def update():
            self.update_counts()
            self.update_events()
            draw()

        def draw():
            for name in self.lines.keys():
                self.lines[name].remove()

            CEIL_X = self.plot_params["ceiling"]*1.5
            np_counts = np.array(extract_var(self.jpg_data, "new_count"))

            self.lines["edited"] = ax.fill_between(
                np.arange(0, self.length) + 0.5, 0, CEIL_X,
                where=extract_var(self.jpg_data, "edited"),
                facecolor="#D8BFAA", alpha=0.5
            )
            self.lines["selected"] = ax.fill_between(
                np.arange(0, self.length) + 0.5, 0, CEIL_X,
                where=extract_var(self.jpg_data, "selected"),
                facecolor="#F00314"
            )
            self.lines["count"] = ax.fill_between(
                range(self.length), 0, np_counts,
                facecolor="black"
            )
            self.lines["threshold"] = ax.axhline(
                self.plot_params["resp_thresh"], color="#14B37D"
            )

            fig.canvas.draw_idle()

        def on_slide(val):
            for key in self.plot_params.keys():
                if key != "ceiling":
                    slider_key = key[:key.find("_")].upper()
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
                    img = cv2.imread(self.jpg_data[n]["filepath"])
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
                    if event.key == "z":
                        self.press = [-1, i]
                    elif event.key == "x":
                        self.press = [1e5, i]
                    elif event.key == ",":
                        self.press = [0, i]
                if event.key in "zxc,":
                    image_pano(event.xdata)
                elif event.key == "v":
                    self.toggle = not self.toggle
                    image_pano(event.xdata)
                elif event.key == ".":
                    self.user_edits = {}

        def off_key(event):
            try:
                if event.xdata is not None and event.key in "zx,":
                    i = int(round(event.xdata))
                    low, high = sorted((self.press[1], i))
                    lo_to_hi = range(max(0, low), min(self.length, high+1))
                    if event.key in "zx":
                        new_edits = {i: self.press[0] for i in lo_to_hi}
                        self.user_edits = {**self.user_edits, **new_edits}
                    elif event.key == ",":
                        for i in lo_to_hi:
                            self.user_edits.pop(i, None)
                self.press = [None, None]
                update()
            except TypeError:
                pass

        plt.rc("font", **{"size": 8})
        plt.rcParams["keymap.back"] = "left, backspace"

        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.canvas.set_window_title("Response Filtering")
        fig.subplots_adjust(0.07, 0.18, 0.97, 0.97)

        ax.grid(alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_ylim([0, self.plot_params["ceiling"]])
        ax.set_xlim([0, min(500, self.length)])

        ax.set_xlabel("Frame")
        ax.set_ylabel("Response")
        plt.yticks(rotation=45)

        for name in ("count", "threshold", "selected", "edited"):
            self.lines[name] = ax.axhline()

        for name, pos, max_val, init, fmt in SLIDER_PARAMS:
            slider_ax = fig.add_axes([0.125, pos, 0.8, 0.02])
            self.sliders[name] = Slider(
                slider_ax, name, 0, max_val,
                valinit=init, valfmt=fmt, color="#003459", alpha=0.5
            )
            self.sliders[name].on_changed(on_slide)

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("key_release_event", off_key)
        fig.canvas.mpl_connect("button_press_event", on_click)

        try:
            ax.fill_between(
                np.arange(0, self.length) + 0.5, 0, CEIL_X,
                where=[x["from_midnight"] < 4.5 for x in self.jpg_data],
                facecolor="#003459", alpha=0.5
            )
        except KeyError:
            pass

        update()
        plt.show()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("1) Please navigate to folder with camera-trapping images.")
    jpg_paths = find_imgs()

    jpg_data = attach_exif(jpg_paths)
    jpg_data.sort(key=lambda x: x["datetime"])

    print("2) Images are being processed.")
    processed_data = process_jpgs(jpg_data)

    for row in processed_data:
        del row["datetime"]

    if type(processed_data) is not tuple:
        cam = Cam(processed_data)

        print("3) Please choose a location for an initial save.")
        cam.save()

        print("4) Use the interactive plot to select images for export.")
        print(GUIDE)
        cam.plot()

        print("5) Save once again, so changes are recorded.")
        cam.save()

        print("6) Finally, choose a location for export.")
        cam.export()
