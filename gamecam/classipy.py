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
        #self.combo.pack(side="bottom")

        #self.combo.bind("<<ComboboxSelected>>", self.combo_updated)

    def combo_updated(self, event):
        self.parent.main.update()


class Main(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.panel1 = None
        self.panel2 = None
        self.index = 0

        self.image_paths = sorted(self.parent.navbar.get_dir())
        self.update()

        self.b = tk.Button(self.parent, text="OK", command=self.update)
        self.b.pack()

    def update(self):
        self.index = min(len(self.image_paths) - 1, max(0, self.index))
        max_size = max(100, self.winfo_width() // 2)
            
        im_paths = (self.image_paths[self.index+i] for i in range(2))
        im_pil = (Image.open(fp) for fp in im_paths)
        im_resize = (pil_resize(max_size, im) for im in im_pil)
        im_tk = [pil_to_tkinter(im) for im in im_resize]

        im1, im2 = im_tk
        if self.panel1 is None or self.panel2 is None:
            self.panel1 = tk.Label(image=im1)
            self.panel2 = tk.Label(image=im2)
            self.panel1.image = im1
            self.panel2.image = im2
            self.panel1.pack(side="left")
            self.panel2.pack(side="right")
        else:
            self.panel1.configure(image=im1)
            self.panel2.configure(image=im2)
            self.panel1.image = im1
            self.panel2.image = im2


class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.HOME_DIR = os.path.expanduser("~")

        self.panel1 = None
        self.panel2 = None
        self.index = 0

        self.image_paths = sorted(self.get_dir())
        self.update()

        self.b = tk.Button(self.parent, text="OK", command=self.update)
        self.b.pack(side="bottom")

    def get_dir(self):
        output = filedialog.askopenfilenames(
            initialdir=self.HOME_DIR ,
            title="Choose files to view",
        )
        return output

    def update(self):
        self.index = min(len(self.image_paths) - 1, max(0, self.index))
        print(self.winfo_width())
        print(self.winfo_reqwidth())
        print()
        print(self.parent.winfo_width())
        print(self.parent.winfo_reqwidth())
        max_size = max(100, self.parent.winfo_width() // 2)

        im_paths = (self.image_paths[self.index+i] for i in range(2))
        im_pil = (Image.open(fp) for fp in im_paths)
        im_resize = (pil_resize(max_size, im) for im in im_pil)
        im_tk = [pil_to_tkinter(im) for im in im_resize]

        im1, im2 = im_tk
        if self.panel1 is None or self.panel2 is None:
            self.panel1 = tk.Label(image=im1)
            self.panel2 = tk.Label(image=im2)
            self.panel1.image = im1
            self.panel2.image = im2
            self.panel1.pack(side="top")
            self.panel2.pack(side="top")
        else:
            self.panel1.configure(image=im1)
            self.panel2.configure(image=im2)
            self.panel1.image = im1
            self.panel2.image = im2


if __name__ == "__main__":
    initialize_root()
