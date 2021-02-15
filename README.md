# tag_tracking
ML based Motion Tracking and Synthetic MR Motion Image Generator

# Summary

`tagsim/` contains code used to generate motion images, and perform a Bloch
simulation to create MR images with proper contrast/features.

`torch_tag/` contains the pyTorch implementation of the tracking network and
code used to train the network.

# tagsim

Software to generate pseudo random deformation MR images.  Includes full image
deformations and cardiac-like deformations, as well as a GPU accelerated Bloch
simulator to generate MR images.

Demo Jupyter notebooks are in the `tagsim/notebooks` folder.

**Installation**: This code requires a C library to be built for gridding.  Running `python setup.py build_ext --inplace` in the tagsim folder should build everything.  If you are using XCode to on Mac for C compiling, replace `setup.py` with `setup_xcode.py` (this disables openMP because stock Mac XCode doesn't support it).

# torch_track

Software containing the neural network for tracking MR images.  The full network implementation and pre-trained network are included, as well as a demo of its usage on an example dataset.  Machine learning is implemented with pyTorch.

Demo Jupyter notebooks are in the `torch_track/notebooks` folder.

A pre-train network for grid-tagged tracking is in the `tagtorch_tracksim/network_saves` folder.

**Installation**: No special installation is required for this software, other than installing required dependencies as they come up.



