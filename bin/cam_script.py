from gamecam import *

home_path = '/Users/user/Data/python/wardle/2AL (35)'

st = time.time()
jpg_paths = find_jpgs(home_path)
end = time.time()
print(f"A\t{end - st}")
#print(jpg_paths[0])

st = time.time()
jpg_data = sorted(attach_exif(jpg_paths), key=lambda x: x['datetime'])
end = time.time()
print(f"B\t{end - st}")
#print(jpg_data[0])

st = time.time()
params = generate_clone_params((882,979,0,203), "right")
processed_data = process_jpgs(jpg_data,
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
