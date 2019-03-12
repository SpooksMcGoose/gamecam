import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="gamecam-sdrabing",
    version="0.1.0",
    author="Shane Drabing",
    author_email="shane.drabing@gmail.com",
    packages=['gamecam', 'gamecam.test'],
    scripts=['scripts/basic_script.py', 'scripts/timed_script.py'],
    url="https://github.com/SpooksMcGoose/gamecam",
    description="Filters out images that don't contain an animal.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
