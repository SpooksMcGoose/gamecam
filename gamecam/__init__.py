import os

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    import Tkinter as tk
    from Tkinter import filedialog

# Work-around for a weird tkinter / matplotlib interaction.
# Note: I would use TkAgg backend, but key bouncing breaks the plot.
root = tk.Tk()
root.withdraw()

name = "gamecam"

__all__ = ["pyrcolate", "classipy"]


def handle_tkinter(mode, init=None):
    """Used for making basic Tkinter dialog calls.

    For some reason, Tkinter doesn"t like to be used within a class,
    so this is here instead.

    Parameters
    ----------
    mode : {"savefile", "openfile", "opendir"}
        Type of file dialog to call.
    init : str, optional
        Starting path for file dialog.

    Returns
    -------
    str
        User-selected path name from file dialog.
    """

    root = tk.Tk()
    root.withdraw()

    if init is None:
        init = os.path.expanduser("~")

    if mode == "savefile":
        output = filedialog.asksaveasfilename(
            initialdir=init,
            filetypes=(("Save files", "*.sav"), ("All files", "*.*"))
        )
    elif mode == "openfile":
        output = filedialog.askopenfilename(
            initialdir=init,
            title="Select file",
            filetypes=(("Save files", "*.sav"), ("All files", "*.*"))
        )
    elif mode == "opendir":
        output = filedialog.askdirectory(
            initialdir=init,
            title="Select folder",
        )

    root.update()
    root.destroy()

    return output


def find_imgs(dirpath=None, img_type=(".jpg", ".jpeg")):
    """Walks directory path, finding all files ending in img_type.

    Parameters
    ----------
    dirpath : str, optional
        If no path provided, the user can navigate to it via tkinter window.
    img_type : tuple, optional
        By default, finds JPG image types, but can be changed if camera
        exports a different filetype.

    Returns
    -------
    list of dictionaries
        Contains filenames and filepaths.
    """

    if dirpath is None:
        dirpath = handle_tkinter("opendir")

    if dirpath:
        output = []

        for dir_, subdir, files in os.walk(dirpath):
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
    else:
        print("NO IMAGE DIRECTORY SPECIFIED!")
        return []


# Why imports at the bottom? These modules use the above functions.
import gamecam.pyrcolate
import gamecam.classipy
