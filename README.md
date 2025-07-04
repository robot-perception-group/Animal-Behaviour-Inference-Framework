<h1 align="center">
  A Framework for Fast, Large-scale, Semi-Automatic Inference of Animal Behavior from Monocular Videos
  <br/>
  <img src="resources/animal_behavior_repo_cover2.png"><br/>
</h1>

<h2 align="center">
Welcome to the repo to perform fast, large-scale, semi-automatic inference of animal behavior from monocular videos.
</h2>

<!---
<div align="center">
  <img src="resources/labelme_screenshot.png" width="70%">
</div>
--->

<br/>

## Description
The animal behavior inference workflow is built on top of the open-source graphical image annotation tool [Smarter-labelme](https://github.com/robot-perception-group/smarter-labelme) with the goal of minimizing the cumbersome and time-intensive 
manual effort associated with generating reliable behavior-dense annotated datasets. The generated dataset can be easily used to train diverse machine learning models as well as to perform behavioral analysis.

## Advantages

- [x] The workflow allows collecting dense behavior data over large time and spatial scales
- [x] Enables researchers to quickly annotate animals and behaviors of interest to generate dense behavior and annotation datasets
- [x] The underlying light-weight network architecture and implementation can be employed for real-time animal detection and behavior classification in the wild
- [x] The workflow is built around videos from consumer-grade single cameras enabling easy adoption for research and conservation in the field

We demonstrate the benefits of using this workflow on aerial video footage of zebras recorded at Mpala in Kenya. The [full annotated dataset](https://keeper.mpdl.mpg.de/d/a9822e000aff4b5391e1/) is freely available for the research community.


A [Preprint outlining the framework](https://doi.org/10.1101/2023.07.31.551177) has been uploaded to bioRxiv and a video abstract can be found [here](https://www.youtube.com/watch?v=Zu-t0JJsz5o).

**To adapt Smarter-labelme for your own project, please follow the guide [here](/howto/README.md)**.

<div align="center">
  <img src="resources/auto_annotate_track.gif">
</div>


## Requirements

- Ubuntu / macOS / Windows
- Python2 / Python3
- [PyQt4 / PyQt5](http://www.riverbankcomputing.co.uk/software/pyqt/intro) / [PySide2](https://wiki.qt.io/PySide2_GettingStarted)
- PyTorch


## Installation
You can install via pip:

```bash
python3 -m pip install --upgrade "git+https://github.com/robot-perception-group/Animal-Behaviour-Inference-Framework@master"
```

### Hint on Pytorch

Pytorch will be installed by pip as a dependency by the above command, if it is not already installed, however you will want to select the matching version for your system from https://pytorch.org/get-started/locally/ -- if you do not have a GPU, use

```
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

## Usage

Run `smarter_labelme --help` for detail.  
The annotations are saved as a [JSON](http://www.json.org/) file.

```bash
smarter_labelme  # just open gui
```

### Known Installation Issues.

Due to a version mismatch between OpenCV and system Qt libraries, on some system the following error might occur when attempting to run smarter-labelme:

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in ".../site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
```

If this happens, you can run the following commands to remove the conflicting opencv qt libraries:

```
python3 -m pip uninstall opencv-python
python3 -m pip install --upgrade opencv-python-headless
```

### Neural Network weights.

Smarter-labelme will automatically download pretrained network weights via torch.hub on the first start. They will be cashed in your local user directory and use approximately 200 Mb of space. You can use your own weights instead with the `--ssdmodel` and `--re3model` flags.

### Command Line Arguments
- `--output` specifies the location that annotations will be written to. Annotations will be stored in this directory with a name that corresponds to the image that the annotation was made on.
- The first time you run labelme, it will create a config file in `~/.labelmerc`. You can edit this file and the changes will be applied the next time that you launch labelme. If you would prefer to use a config file from another location, you can specify this file with the `--config` flag.
- Without the `--nosortlabels` flag, the program will list labels in alphabetical order. When the program is run with this flag, it will display labels in the order that they are provided.
- `--labels` allows to limit labels to a determined set, for example MSCOCO. The parameter can be a text file with one label per line or a comma-separated list.
- `--flags` allows to specify per-image flags. The parameter can be a text file with one label per line or a comma-separated list.
- `--labelflags` allows to specify per-annotation flags to give additional information beyond the label, for example for behavior annotation. The syntax is JSON with regular expressions, for example `--labelflags {.*: [occluded,running,walking,sleeping]}`. There is one internal labelflag - "disable_visual_tracking" which can be used to disable the automated visual tracker. Objects with this flag set will simply be copied to the next frame unchanged whenever `Track-Polygon` is engaged.

## Video annotation procedure with instance tracking.

1. Install Smarter-labelme
2. If you have a .AVI or .MP4 file, use `ffmpeg` to extract the video. Smarter-labelme provides a wrapper to preserve frame IDs `smarter_labelme_video2frames <video> <output_folder> [--fps [fps]|"full"]`. The default fps is 8, specify "full" to extract all frames from the video.
3. Start `smarter_labelme` with appropriate labelflags for your task (see above). e.g. `smarter_labelme --labelflags '{.*: ["grazing","standing","walking","running"]}'`
4. Open the directory where your extracted the video frames. They will be displayed in order, sorted by filename.
5. You can try to annotate with the "Auto-annotate" button. Each detected object will receive a label based on detected class and a unique ID.
6. Fix any misdetections and/or add not detected objects. The shortcut for rectangle annotations is `Ctrl+R`. Press `ESC` to go back to edit mode when you are done.
7. All objects should have unique labels. This is easiest achieved if giving the type of object, followed by a unique number.
8. Enter Edit mode (`ESC`), then select those objects you would like to track across frames. You can do so by clicking the first entry in the `Polygon-Labels` widget and then shift-clicking the last.
9. Click `Track-Polygon` to engage the tracker for the selected objects.
10. Switch to the next frame. Smarter-labelme will automatically try to track and follow all selected object instances and add them to the next frame.
11. Fix all bounding boxes as needed. If the tracker consistently fails or misbehaves for an object, you can edit this object-label (`Ctrl+E`) and select the "disable_visual_tracking" flag.
12. Continue this process for all frames. If new objects appear in the video, add new bounding boxes, then repeat from step 8 on that frame. Objects that disappeared can simply be deleted.
13. The Annotations will be stored in machine readable JSON files in the `Annotations` subfolder of your input directory.

## Acknowledgement

This repo is the fork of [bhavyaajani/smart-labelme](https://github.com/bhavyaajani/smart-labelme).


## Cite This Project

If you use this project in your research or wish to refer to the baseline results published in the README, please use the following BibTeX entries.

```bash
@article{Price_et_al_2025,
	author = {Eric Price and Pranav C. Khandelwal and Daniel I. Rubenstein and Aamir Ahmad},
	title = {A Framework for Fast, Large-scale, Semi-Automatic Inference of Animal Behavior from Monocular Videos},
	journal = {Methods in Ecology and Evolution},
	volume = {},
	number = {},
	pages = {},
	keywords = {Animal behavior Inference, Drone based monitoring, Deep neural network based inference, Automatic labeling, Labelled aerial datasets},
	doi = {},
	url = {[https://besjournals.onlinelibrary.wiley.com/doi/abs/10.1111/2041-210X.70056](https://www.biorxiv.org/content/early/2023/08/02/2023.07.31.551177)},
	eprint = {},
	year = {2025},
	comments={Accepted, 1st July 2025}
}

@article{Price2023.07.31.551177,
	author = {Eric Price and Pranav C. Khandelwal and Daniel I. Rubenstein and Aamir Ahmad},
	title = {A Framework for Fast, Large-scale, Semi-Automatic Inference of Animal Behavior from Monocular Videos},
	elocation-id = {2023.07.31.551177},
	year = {2023},
	doi = {10.1101/2023.07.31.551177},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/08/02/2023.07.31.551177},
	eprint = {https://www.biorxiv.org/content/early/2023/08/02/2023.07.31.551177.full.pdf},
	journal = {bioRxiv}
}

@InProceedings{10.1007/978-3-031-44981-9_12,
   author="Price, Eric and Ahmad, Aamir",
   editor="Lee, Soon-Geul et al.",
   title="Accelerated Video Annotation Driven by Deep Detector and Tracker",
   booktitle="Intelligent Autonomous Systems 18",
   year="2024",
   publisher="Springer Nature Switzerland",
   address="Cham",
   pages="141--153",
   doi = {10.1007/978-3-031-44981-9_12},
   isbn="978-3-031-44981-9"
}
```
```
