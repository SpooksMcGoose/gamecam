# gamecam
Filter out trail-camera images that don't contain an animal (eventually will be able to label exported images as well).

## Motivation
While working for the [Francis Ecology Lab](https://francisecologylab.wixsite.com/francislab), a problem was presented to me. In 2018, there were 800,000+ remote camera images collected for a single experiment. The majority turned out to be false triggers - photos of grass and bushes swaying.

Instead of going through them one by one, I developed this program. Once coded, it only took a week to generate a data table with all the images that contained animals, labeled to genus or species, and that had detection numbers corresponding to one of two common schemes. The data table can be converted into observational units of detections instead of individual images.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install gamecam.

```bash
pip install gamecam-sdrabing
```

## Usage
```python
from gamecam import pyrcolate as pyr

# creates data table from folder
jpg_data = pyr.construct_jpg_data(pyr.input_directory())

# Cam objects store all relevant metadata
cam = pyr.Cam(jpg_data)

help(Cam.plot)                     # shows quick-guide for key bindings
cam.plot()                         # interactive plot for image filtering
cam.save(pyr.input_filename())     # creates a .sav file that can be loaded
cam.export(pyr.input_directory())  # exports images and a .csv file to folder
```

## To-Do
Finish gamecam.classipy, which will allow for rapid labeling of exported images.

## Contributing
This is my first project, so I have no idea what I'm doing.
If you have suggestions or tips, please tell me!

## License
[MIT](https://choosealicense.com/licenses/mit/) © Shane Drabing 2019