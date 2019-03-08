import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="gamecam-spooksmcgoose",
    version="0.0.1",
    author="Shane Drabing",
    author_email="shane.drabing@gmail.com",
    description="Filters out images that don't contain an animal.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpooksMcGoose/gamecam",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
