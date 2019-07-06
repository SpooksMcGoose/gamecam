import json
import operator
import os
import shutil
import sys
import time

import cv2
import exifread as er
import numpy as np

from sklearn.cluster import KMeans
from datetime import datetime as dt, timedelta as td

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, RectangleSelector
import matplotlib.gridspec as gridspec


# CONSTANTS

BIG_NUM = int(1e9)
NEG_NUM = -0.1

RESP_NUM = 3.5


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

COL_ORDER = [
    "old_name", "old_path", "filename", "filepath", "shape",
    "datetime", "24hr", "is_night", "timedelta", "td_minutes",
    "median", "med_diff", "count", "new_count",
    "selected", "user_edit", "trans_edit"
]

RS_PARAMS = {
    "drawtype": 'box', "useblit": True, "button": [1, 3], "minspanx": 5,
    "minspany": 5, "spancoords": "pixels", "interactive": True
}


# GENERAL FUNCTIONS


class Mem():
    def __init__(self):
        pass


def sort_cols(var_name):
    if var_name in COL_ORDER:
        return COL_ORDER.index(var_name)
    else:
        return BIG_NUM


def input_filename():
    while True:
        filename = input("> ")
        try:
            with open(filename, 'w') as _:
                return filename
        except OSError:
            print("File already exists or is invalid, try again.")


def input_directory():
    while True:
        directory = input("> ")
        if os.path.isdir(directory):
            return directory
        else:
            print("Invalid directory path, try again.")


def csv_safe(obj):
    """Puts quotes around strings with commas in them.
    """

    string = str(obj)
    return '"' + string + '"' if "," in string else string


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
            d[k] = str(v).rjust(2, "0")
    return fmt.format(**d)


# SPECIFIC FUNCTIONS (ALL BELOW)


def find_imgs(dirpath, img_type=(".jpg", ".jpeg")):
    """Walks directory path, finding all files ending in img_type.

    Parameters
    ----------
    dirpath : str
        Path to an image-containing directory.
    img_type : tuple, optional
        By default, finds JPG image types, but can be changed if camera
        exports a different filetype.

    Returns
    -------
    list of dictionaries
        Contains filenames and filepaths.
    """

    output = []

    for dir_, _, files in os.walk(dirpath):
        if "_selected" not in dir_:
            found = (
                f for f in files
                if f.lower().endswith(img_type)
            )
            for filename in found:
                filepath = os.path.join(dir_, filename)

                output.append({
                    "filename": filename,
                    "filepath": filepath
                })

    return output


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
    mults = np.array(DIRECTION_MULTS[fill_from[0].upper()])

    a, b, c, d = clone_to
    h, w = b - a, d - c
    h_or_w = np.array([h, h, w, w])

    clone_from = clone_to + (h_or_w * mults)
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

    (a, b, c, d), (e, f, g, h) = clone
    image[a:b, c:d] = image[e:f, g:h]

    image = CROPPER(image, crop)
    return cv2.equalizeHist(image)


def crop_clone_preview(image):
    """Interactive viewer to quickly get crop and clone parameters.

    Parameters
    ----------
    image : numpy.ndarray
        Image array, typically from cv2.imread().

    Returns
    -------
    tuple
        Format is (crop, clone_to, clone_directs). The crop_tuple
        variable can be fed directly into process_jpgs(). Then, use
        "generate_clone_tuples(clone_to, directs[0])" to get the clone
        parameter, or any other direction in the directs list (they've all
        been checked for bounds, unlike generate_clone_tuples).
    """

    mem = Mem()
    mem.crop_tuple = False
    mem.clone_tuple = False
    mem.clone_directs = False

    def update(event):
        if crop_RS.active:
            crop_RS.update()
        if clone_RS.active:
            clone_RS.update()

    def safe_corners(geometry):
        min_y = max(0, min(geometry[0, :]))
        max_y = min(H, max(geometry[0, :]))
        min_x = max(0, min(geometry[1, :]))
        max_x = min(W, max(geometry[1, :]))

        ignore = (min_y > H or max_y < 0 or min_x > W or max_x < 0)
        return tuple(map(
            lambda x: int(round(x)), (min_y, max_y, min_x, max_x))), ignore

    def RS_event(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        cr_corn, cr_ig = safe_corners(crop_RS.geometry)
        cl_corn, cl_ig = safe_corners(clone_RS.geometry)

        mod = image.copy()

        if not cl_ig:
            mem.clone_tuple = cl_corn

            trials = []
            for direct in ("A", "B", "R", "L"):
                tups = generate_clone_tuples(cl_corn, direct)
                a, b, c, d = tups[1]
                if not (a < 0 or b > H or c < 0 or d > W):
                    trials.append((tups, direct))

            choose = []
            for (tups, direct) in trials:
                test = image.copy()

                (a, b, c, d), (e, f, g, h) = tups
                test[a:b, c:d] = test[e:f, g:h]

                diff = cv2.absdiff(test, image)
                choose.append((sum(cv2.sumElems(diff)), direct, test))
            choose.sort()

            if choose:
                _, directs, imgs = list(zip(*choose))
                mem.clone_directs = directs
                mod = imgs[0]

            del choose
        else:
            mem.clone_tuple = False
            mem.clone_directs = False

        if cr_ig:
            cr_corn = (0, H, 0, W)
            mem.crop_tuple = False
        else:
            mem.crop_tuple = cr_corn

        y1, y2, x1, x2 = cr_corn
        mod = mod[y1:y2, x1:x2]

        mod = cv2.cvtColor(mod, cv2.COLOR_BGR2RGB)
        ax_mod.imshow(mod)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    H, W, *_ = image.shape
    scale = 10
    lo_W, hi_W = (0 - W // scale), (W + W // scale)
    lo_H, hi_H = (0 - H // scale), (H + H // scale)

    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.25])

    fig = plt.figure()
    ax_crop = fig.add_subplot(gs[0, 0])
    ax_clone = fig.add_subplot(gs[0, 1])
    ax_mod = fig.add_subplot(gs[1, :])

    crop_RS = RectangleSelector(
        ax_crop, RS_event, **RS_PARAMS)
    clone_RS = RectangleSelector(
        ax_clone, RS_event, **RS_PARAMS)

    ax_crop.set_title("Crop")
    ax_clone.set_title("Clone")
    ax_mod.set_title("Result")

    for axe in (ax_crop, ax_clone):
        axe.set_xticklabels([])
        axe.set_yticklabels([])
        axe.set_xlim(lo_W, hi_W)
        axe.set_ylim(hi_H, lo_H)

    ax_crop.imshow(rgb)
    ax_clone.imshow(rgb)
    ax_mod.imshow(rgb)

    plt.connect('draw_event', update)
    plt.show()

    return mem.crop_tuple, mem.clone_tuple, mem.clone_directs


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
        cv2.THRESH_BINARY)
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
        cv2.THRESH_BINARY)
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
        cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

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
            remain = strfdelta(td(seconds=total - elapsed),
                               "{days}:{hours}:{minutes}:{seconds}")
            print(f"{progress}% done. {remain} left.")

        jpg = cv2.imread(row["filepath"], 0)
        curr = preprocess(jpg, crop, clone)

        row["shape"] = jpg.shape
        row["median"] = hist_median(jpg)

        if not thresh_init:
            threshold = row["median"]*1.05

        try:
            row["count"] = method(curr, prev, threshold, ksize, min_area)
        except cv2.error as inst:
            if "(-209:Sizes" in str(inst):
                (a, b), (c, d) = curr.shape[:2], prev.shape[:2]
                h, w = min(a, c), min(b, d)
                tup = (0, h, 0, w)
                print("FUNCTION ABORTED!\n"
                      "Not all images are of same size, "
                      "consider using the crop parameter.\n"
                      f"Try crop={tup}.")
                return tup
            else:
                print(inst)

        prev = curr
        output.append(row)

    return output


def construct_jpg_data(
    dirpath,
    parse_tags=DEFAULT_PARSE,
    sort_key=SORT_BY_DT,
    process_options={}
):
    """Performs all necessary steps to make jpg_data feedable to Cam().

    Parameters
    ----------
    dirpath : str
        Path to an image-containing directory.
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

    jpg_paths = find_imgs(dirpath)
    print(f"{len(jpg_paths)} images found.")
    jpg_data = attach_exif(jpg_paths, parse_tags)
    jpg_data.sort(key=sort_key)
    if process_options == {}:
        print("Crop and clone image as desired.")
        crop, clone_to, directs = crop_clone_preview(
            cv2.imread(jpg_data[len(jpg_data) // 2]["filepath"]))
        clone = (generate_clone_tuples(clone_to, directs[0]) if clone_to
                 else False)
        process_options = {"crop": crop, "clone": clone}
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
    export(directory)
        Exports selected images and a .csv file.
    load(filename)
        Loads a .sav file.
    mark_edits(i)
        Marks which photos have been edited.
    plot()
        Interactive plot used to select images for export.
    save(filename)
        Dumps a JSON object as a .sav file.
    update_counts()
        Updates jpg_data attribute with new counts.
    update_events()
        Updates jpg_data attribute with new events.
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

        self.plt_vals = DEFAULT_PLOT_PARAMS

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

            h, w, *_ = cv2.imread(jpg_data[0]["filepath"]).shape

            self.plt_vals["ceiling"] = h * w * 0.02
            self.plt_vals["resp_thresh"] = (
                (self.plt_vals["ceiling"] / 2) ** (1 / RESP_NUM))

            self.attach_diffs("median", "med_diff")

            for row in self.jpg_data:
                row["med_diff"] = abs(row["med_diff"])
                row["selected"] = False
                row["user_edit"] = False
                row["trans_edit"] = False

            if self.dt_present:
                self.attach_diffs("datetime", "timedelta")
                for row in self.jpg_data:
                    row["24hr"] = to_24_hour(row["datetime"])
                    row["td_minutes"] = round(
                        row["timedelta"].total_seconds() / 60, 2)

                day_hr = extract_var(self.jpg_data, "24hr")
                # meds = extract_var(self.jpg_data, "median")

                X = np.array(day_hr).reshape(-1, 1)
                kmeans = KMeans(n_clusters=3).fit(X)

                night_indices = [np.argmin(kmeans.cluster_centers_),
                                 np.argmax(kmeans.cluster_centers_)]

                for n, index in enumerate(kmeans.labels_):
                    self.jpg_data[n]["is_night"] = index in night_indices

    def save(self, filename):
        """Dumps a JSON object with jpg_data, plot_params, and user_edits.
        """

        temp_data = [
            {k: v for k, v in row.items()}
            for row in self.jpg_data]
        for row in temp_data:
            if "datetime" in row.keys():
                row["datetime"] = dt.strftime(
                    row["datetime"], "%Y-%m-%d %H:%M:%S")
            if "timedelta" in row.keys():
                row["timedelta"] = row["timedelta"].total_seconds()
            if "selected" in row.keys():
                row["selected"] = int(row["selected"])

        with open(filename, "w") as f:
            f.write(json.dumps(self.plt_vals) + "\n")
            f.write(json.dumps(temp_data) + "\n")
            f.write(json.dumps(self.user_edits))

    def load(self, filename):
        """Loads a .sav file, the Cam() object is now identical to the last.
        """

        with open(filename, "r") as f:
            self.plt_vals = json.loads(next(f))
            self.plt_vals["resp_thresh"] = (
                self.plt_vals["resp_thresh"] ** (1 / RESP_NUM))
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

    def export(self, directory):
        """Exports selected images and a .csv file to specified directory.
        """

        write_data = []

        for i, row in enumerate(self.jpg_data):
            write_row = row.copy()
            if row["selected"]:
                if self.dt_present:
                    dt_ISO = dt.strftime(
                        row["datetime"], "%Y%m%dT%H%M%S"
                    )
                else:
                    dt_ISO = str(i)
                new_name = "_".join((dt_ISO, row["filename"]))
                new_path = os.path.join(directory, new_name)

                write_row["filename"] = new_name
                write_row["filepath"] = new_path
                write_row["old_name"] = row["filename"]
                write_row["old_path"] = row["filepath"]

                shutil.copy2(row["filepath"], new_path)
                write_data.append(write_row)
        if write_data:
            with open(os.path.join(directory, "_export.csv"), "w") as f:
                variables = sorted(write_data[0].keys(),
                                   key=sort_cols)
                for i, row in enumerate(write_data):
                    if i != 0:
                        f.write("\n")
                    else:
                        f.write(",".join(variables) + "\n")
                    f.write(",".join(csv_safe(row[v]) for v in variables))
        else:
            raise ValueError("No images selected for export.")

    def attach_diffs(self, var, new_var):
        """Finds the difference between the current and previous variable.

        Requires a list of dictionaries.
        """

        prev = self.jpg_data[0][var]
        for row in self.jpg_data:
            curr = row[var]
            row[new_var] = curr - prev
            prev = curr

    def mark_edits(self, i, kind):
        """Marks which photos to label as edited based on [i]ndex.
        """

        if i == 0:
            self.jpg_data[i][kind] = True
        else:
            self.jpg_data[i][kind] = True
            self.jpg_data[i-1][kind] = True

    def update_counts(self):
        """Updates jpg_data with new counts (response metric).

        Variable new_count is attached to jpg_data. Day to night transitions
        are filtered out based on slider. Counts taken at with nighttime are
        multiplied based on slider. Manual user edits are applied.
        """

        for i, row in enumerate(self.jpg_data):
            row["user_edit"] = False
            row["trans_edit"] = False

            new_count = row["count"]
            if row["med_diff"] > self.plt_vals["trans_thresh"]:
                new_count = 0
                self.mark_edits(i, "trans_edit")
            if self.dt_present:
                new_count *= 1 + (
                    self.plt_vals["night_mult"] * row["is_night"])

            row["new_count"] = new_count

        for i, shift in self.user_edits.items():
            self.jpg_data[i]["new_count"] = shift
            self.mark_edits(i, "user_edit")

    def update_events(self):
        """Updates jpg_data with new events (runs of detection).

        An event is a contiguous sequence of images. First, images are
        identified as being selected or not. Then, events are lumped based
        on time since last image (SMOOTH slider value).
        """

        prev = self.jpg_data[0]
        for i, curr in enumerate(self.jpg_data):
            prev["selected"] = (
                prev["new_count"] > self.plt_vals["resp_thresh"] or
                curr["new_count"] > self.plt_vals["resp_thresh"])

            if i == self.length-1:
                curr["selected"] = (
                    curr["new_count"] > self.plt_vals["resp_thresh"])
            prev = curr

        if self.dt_present:
            for move in (1, -1):
                is_neg = move < 0
                prev = self.jpg_data[-is_neg]
                for i in range(-is_neg, (self.length * move) - is_neg, move):
                    curr = self.jpg_data[i]
                    boo = (not curr["selected"] and prev["selected"] and
                           not (prev["user_edit"] and curr["user_edit"]))
                    if boo:
                        if not is_neg:
                            lapse = curr["td_minutes"]
                        else:
                            lapse = prev["td_minutes"]

                        curr["selected"] = (
                            lapse <= self.plt_vals["smooth_time"])
                    prev = curr
        else:
            nudge = int(self.plt_vals["smooth_time"])
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
           x - Hold and release to INCREASE response value to Inf.
           z - Hold and release to DECREASE response value to 0.
           , - Hold and release to REMOVE EDITS.
           . - Press to RESET ALL EDITS.
           v - Press for EQUALIZED image (for dark images).
        """

        try:
            self.jpg_data
        except AttributeError as inst:
            raise inst

        SLIDER_PARAMS = [
            ("RESP", 0.08, 0, 100, self.plt_vals["resp_thresh"], "%.2e"),
            ("TRANS", 0.06, 0, 120, self.plt_vals["trans_thresh"], "%.1f"),
            ("SMOOTH", 0.04, 0, 10, self.plt_vals["smooth_time"], "%.1f"),
            ("NIGHT", 0.02, -1, 50, self.plt_vals["night_mult"], "%.1f")
        ]

        def update():
            self.update_counts()
            self.update_events()
            draw()

        def draw():
            for name in self.lines.keys():
                self.lines[name].remove()

            np_counts = np.array(extract_var(self.jpg_data, "new_count"))

            self.lines["edited"] = ax.fill_between(
                np.arange(0, self.length) + 0.5, -BIG_NUM, BIG_NUM,
                where=[r["trans_edit"] or r["user_edit"]
                       for r in self.jpg_data],
                facecolor="#D8BFAA", alpha=0.5)
            self.lines["selected"] = ax.fill_between(
                np.arange(0, self.length) + 0.5, -BIG_NUM, BIG_NUM,
                where=extract_var(self.jpg_data, "selected"),
                facecolor="#F00314")
            self.lines["count"] = ax.fill_between(
                range(self.length), 0, np_counts, facecolor="black")
            self.lines["threshold"] = ax.axhline(
                self.plt_vals["resp_thresh"], color="#14B37D")

            fig.canvas.draw_idle()

        def on_slide(val):
            for key in self.plt_vals.keys():
                if key != "ceiling":
                    slider_key = key[:key.find("_")].upper()
                    if key == "resp_thresh":
                        mod = self.sliders[slider_key].val ** RESP_NUM
                        self.plt_vals[key] = mod
                        self.sliders[slider_key].valtext.set_text("%.2e" % mod)
                    else:
                        self.plt_vals[key] = self.sliders[slider_key].val
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

            pano = np.hstack([img[:min_y, :] for img in stack])
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
                        self.press = [NEG_NUM, i]
                    elif event.key == "x":
                        self.press = [BIG_NUM, i]
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
        ax.set_ylim([0, self.plt_vals["ceiling"]])
        ax.set_xlim([0, min(500, self.length)])

        ax.set_xlabel("Frame")
        ax.set_ylabel("Response")
        plt.yticks(rotation=45)

        for name in ("count", "threshold", "selected", "edited"):
            self.lines[name] = ax.axhline()

        for name, pos, minv, maxv, init, fmt in SLIDER_PARAMS:
            slider_ax = fig.add_axes([0.125, pos, 0.8, 0.02])
            self.sliders[name] = Slider(
                slider_ax, name, minv, maxv,
                valinit=init, valfmt=fmt, color="#003459", alpha=0.5)
            self.sliders[name].on_changed(on_slide)

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("key_release_event", off_key)
        fig.canvas.mpl_connect("button_press_event", on_click)

        ax.fill_between(
            np.arange(0, self.length) + 0.5, -BIG_NUM, 0,
            facecolor="white", alpha=0.75, zorder=2)
        try:
            ax.fill_between(
                np.arange(0, self.length) + 0.5, -BIG_NUM, BIG_NUM,
                where=extract_var(self.jpg_data, "is_night"),
                facecolor="#003459", alpha=0.5)
        except KeyError:
            pass

        on_slide(1)
        plt.show()

        cv2.destroyAllWindows()


if __name__ == "__main__" and 1:
    homedir = sys.path[-1]

    print("→ Please input a directory path with camera-trapping images.")
    jpg_paths = find_imgs(input_directory())

    jpg_data = attach_exif(jpg_paths)
    jpg_data.sort(key=lambda x: x["datetime"])

    print("→ Crop and clone out any timestamps from images.")
    crop, clone_to, directs = crop_clone_preview(
        cv2.imread(jpg_data[len(jpg_data) // 2]["filepath"]))
    clone = generate_clone_tuples(clone_to, directs[0]) if clone_to else False

    print("→ Images are being processed.")
    processed_data = process_jpgs(jpg_data, crop=crop, clone=clone)

    if type(processed_data) is not tuple:
        cam = Cam(processed_data)

        print("→ Please input a file path for an initial save.")
        cam.save(input_filename())

        print("→ Use the interactive plot to select images for export.")
        help(Cam.plot)
        cam.plot()

        print("→ Save once again, so changes are recorded.")
        cam.save(input_filename())

        print("→ Finally, choose a location for export.")
        cam.export(input_directory())
