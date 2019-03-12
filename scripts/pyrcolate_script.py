from gamecam.gamecam import *
import sys

cwd = sys.path[0]
home_path = cwd[:cwd.rfind('/')] + '/data/images'

jpg_paths = find_jpgs(home_path)

jpg_data = attach_exif(jpg_paths)
jpg_data.sort(key=lambda x: x['datetime'])

params = generate_clone_params((882,979,0,203), "right")

processed_data = process_jpgs(
    jpg_data, crop=(0,100,0,0), clone_params=params
)

cam = Cam(processed_data)

cam.plot()

cam.save()

wam = Cam()

wam.load()

wam.plot()

wam.export()
