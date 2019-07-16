import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="gamecam-sdrabing",
    version="0.8.3",
    author="Shane Drabing",
    author_email="shane.drabing@gmail.com",
    packages=setuptools.find_packages(),
    url="https://github.com/shanedrabing/gamecam",
    description="Remote-camera software suite.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    data_files=[
        ("", ["LICENSE.txt"])
    ],
    install_requires=[
        "opencv-python", "piexif", "numpy", "matplotlib", "scikit-learn"
    ]
)
