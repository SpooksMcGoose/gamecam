import os
import cv2
from PIL import ImageTk
from PIL import Image

try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import filedialog
except ImportError:
    import Tkinter as tk
    from Tkinter import ttk
    from Tkinter import filedialog


# GENERAL FUNCTIONS


def initialize_root():
    root = tk.Tk()
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()


# IMAGE FUNCTIONS


def opencv_to_tkinter(cv_im):
    cv_im = cv2.cvtColor(cv_im, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv_im)
    return pil_to_tkinter(pil_im)


def pil_to_tkinter(pil_im):
    return ImageTk.PhotoImage(pil_im)


def pil_resize(max_size, pil_im):
    max_size = int(max_size)
    w, h = pil_im.size
    scale = min(max_size / w, max_size / h)
    return pil_im.resize((round(w * scale), round(h * scale)), Image.ANTIALIAS)


# GUI CLASSES


class Navbar(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.HOME_DIR = os.path.expanduser("~")

        self.combo = ttk.Combobox(self.parent)
        self.combo['values']= (240, 480, 600, 768, 1024)
        self.combo.current(0)
        self.combo.pack(side="top")

        self.combo.bind("<<ComboboxSelected>>", self.combo_updated)

    def combo_updated(self, *args):
        self.parent.main.update()

    def get_dir(self):
        output = filedialog.askopenfilenames(
            initialdir=self.HOME_DIR ,
            title="Choose files to view",
        )
        return output


class Main(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.panel1 = None
        self.panel2 = None
        self.index = 0

        self.image_paths = sorted(self.parent.navbar.get_dir())
        self.update()

    def update(self):
        self.index = min(len(self.image_paths) - 1, max(0, self.index))
        combo_val = self.parent.navbar.combo.get()

        im_paths = (self.image_paths[self.index+i] for i in range(2))
        im_pil = (Image.open(fp) for fp in im_paths)
        im_resize = (pil_resize(combo_val, im) for im in im_pil)
        im_tk = [pil_to_tkinter(im) for im in im_resize]

        im1, im2 = im_tk
        if self.panel1 is None or self.panel2 is None:
            self.panel1 = tk.Label(image=im1)
            self.panel2 = tk.Label(image=im2)
            self.panel1.image = im1
            self.panel2.image = im2
            self.panel1.pack(side="right", padx=10, pady=10)
            self.panel2.pack(side="right", padx=10, pady=10)
        else:
            self.panel1.configure(image=im1)
            self.panel2.configure(image=im2)
            self.panel1.image = im1
            self.panel2.image = im2


class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.navbar = Navbar(self)
        self.main = Main(self)

        self.navbar.pack(side="bottom", fill="y", expand=True)
        self.main.pack(side="top", fill="y")


if __name__ == "__main__":
    initialize_root()
