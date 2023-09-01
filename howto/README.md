<h1 align="center">
  Adapting the Smarter-labelme workflow for your project
  <br/>
  <img src="../resources/workflow_v1.svg"><br/>
</h1>

The entire workflow is divided into 3 streams (S1, S2, and S3). The workflow is initiated from S1 by manually annotating pixel-exact bounding boxes of animals to generate animal detector training data followed by training the detector network. The trained animal detector is used in S2 to perform manual annotation of behaviors and generate a training dataset to train the behavior classifier. S3 takes the output of S2 to semi-automate the entire behavior annotation process and rapidly generate more training data to fine-tune the behavior classifier and reach sufficient animal detection and behavior inference accuracy to automate the entire annotation process or directly use the annoations for behavior analysis.

**Before following the steps below, please install smarter-labelme. Instructions to install can be found [here](/README.md).**

The process below is described for a situation where there are multiple individuals in a video. The same process can be applied for videos with a single individual which will make the annoation process even faster.

## Stream 1 (S1)

**Goal**: Generate pixel-exact bounding boxes to detect animal(s) of interest. **We do not worry about the behavior in S1. It will be addressed in S2**

1. Extract frames from video using the command: \
```smarter_labelme_video2frames <video> <output_folder> [--fps [fps]|"full"]``` \
Choose the `fps` value such that there aren't any jumps in the individual's position. For example, slow movements can have a low `fps` and fast movements should have frames extracted at high `fps`.\
Note: The extracted frames will be saved in the `<output_folder>` which would be opened in smarter-labelme for annotations.
2. Start smarter-labelme and load the `<output_folder>` using the `Open DIR` option.
3. Navigate to a frame that has the majority of the individuals clearly visible. This would be your starting frame. You can either use `AutoAnnotate` or manually draw bounding boxes around each animal using the `Create Rectangle` option. The bounding boxes will be populated in the `Polygon Labels` section.\
Note: Using `AutoAnnotate` automatically assigns an identifier in the following format: `animal_uniqueID {behavior}`. **For S1 we do not worry about the `behavior` identified**.
4. Once all the individuals are annotated, select all the `Polygon Labels` and then click `Track Polygon`. Now when you move one frame forward or backward all the annotated bounding boxes will be updated in the next frame. You should check if all bounding boxes are correct, if not, they should be adjusted before proceeding to the next frame.\
Note: The `Track Polygon` works when moving forward and backward a frame.

If required, repeat steps 1-4 on other videos such that the annotated dataset captures variety in animal poses, behaviors, lightning conditions, and landscapes.

### Training the detector network

Once sufficient training data has been annotated, you can train Smarter-labelme to identify and track your animals of interest.\
Note: You can get a quick overview of your training data including number of annotations, labels, and other such statistics by running the following command: \
```python3 analyse_dataset.py <smarter-labelme-image-folder> [<more folders>]```

To train Smarter-labelme: \
1. ```python3 make_combined_dataset.py <smarter-labelme-image-folder> [<more image folders] <destination folder>```\
This will create a training dataset from one or multiple annotated datasets combined with MSCOCO 2017
2. Make a folder to save the training snapshots: \ 
```mkdir snapshots```\
Train the netowork: \
```./train.sh <dataset_folder> snapshots``` \
This will train SSD Multibox with parameters suited for smarter-labelme usage and store snapshots in snapshot folder

Once the training is complete, you can move on to Stream 2

## Stream 2 (S2)

**Goal**: Leverage the animal detector to rapidly classify desired behaviors.\
Note: At this point the detector should be able to track and generate bounding boxes for all animals of interest in each frame.

1. Load Smarter-labelme with the trained weights from S1:
```smarter_labelme --labelflags '{animal: ["behavior1","behavior2","behavior3","behavior4"]}' --ssdmodel ssd_animal.pt```\
Note: In the above command, 'animal', 'behavior*', and 'ssd_animal' are place holders which should be replaced by you. The behaviors specified here will be the ones available to choose within Smarter-labelme
2. Reopen the annotation folders that were used in S1 to rapidly generate bounding boxes for each animal in each frame. Ideally, this can now happen with you just moving frames forward or backward and not requiring manual input the correct the bounding boxes. Once bounding boxes are assigned for each animal across all frames, you can move to the next step. \
Note: An important distinction between S1 and S2 is that in S2 the bounding boxes produced by the animal detector are not required to be pixel-exact boxes. A more relaxed bounding box constraint further reduces the behavior annotation time while making the trained behavior classifier more robust to variations in bounding box positions with respect to the animal.
3. When bounding boxes exist for an individual across consecutive frames, group frame selection can be performed inside Smarter-labelme to change the behavior of the animal across all selected frames simultaneously. You can use this feature by identifying the start and end frame for a particular behavior for an individual, select all frames in between, and then edit the `polygon label` to update all frames to the same behavior.\
This process can be quickly performed for each individual in the video to generate the behavior annotations.
4. Repeat steps 1 to 3 for all other videos ensuring a comparable number of annotations for each behavior that capture different lighting conditions, poses, and individuals.

### Training the behavior classifier
To get a quick overview of the annotated dataset, you can use the same analyse dataset command as in S1:\
```python3 analyse_dataset.py <smarter-labelme-image-folder> [<more folders>]```
1. Make the classifier training data with the following command:\
```python3 make_training_data.py <smarter-labelme-image-folder> [<more image folders] <destination folder>```\
Creates a classifier training dataset from one or multiple annotated datasets. It stores different datasets in destination_folder/train destination_folder/val and destination_folder/test
2. Make a folder to save the training snapshots: \ 
```mkdir snapshots```\
Train the netowork: \
```python3 train.py --traindata <training_data_folder> --testdata <validation_data_folder> --save-model-directoy snapshots``` \
Trains a new network with default seed (see --help for other options) and stores it in folder "snapshots"
3. You can check the performance of the trained classifier network on the test data with the following command: \
```python3 test.py --testdata <validation_data_folder> <snapshot>``` \
Tests a specific trained network on the designated test data - see --help for additional options
4. You can also visualize the 'attention' of the network for identifying different behaviors by using a heatmap:
4.1. Make a folder to save the visualizations: \
```mkdir heatmapfolder```
4.2. Create heatmap: \
```python3 heatmaptest.py --testdata <validation_data_folder> <snapshot> <heatmapfolder>``` \
The heatmaps are saved in the 'heatmapfolder'

**At this point in the process, you should have a trained animal detector and behavior classifier which can now be rapidly fine-tuned by adding more relevant training data**

## Stream 3 (S3)
**Goal**: Bootstrapping the entire process for fast animal tracking and behavior annotation to generate more training data from new videos for S2. Using the trained animal detector and behavior classifier from S2, Smarter-labelme tracks the bounding boxes and corresponding behaviors for all animals of interest in the frame.

1. Load Smarter-labelme with the trained animal detector and behavior classifier: \
```smarter_labelme --labelflags '{animal: ["behavior1","behavior2","behavior3","behavior4"]}' --ssdmodel ssd_animal.pt --flagmodel network_classifier.pt```\
Note: In the above command, 'animal', 'behavior*', 'ssd_animal', and 'network_classifier' are place holders which should be replaced by you. The behaviors specified here will be the ones available to choose within Smarter-labelme
2. The process here is similar to S2. A first pass of correcting all the bounding boxes and not the misclassified behaviors is performed. Once bounding boxes have been checked and/or corrected, the behavior of each animal is checked. Starting from the frame where a particular behavior starts, consecutive frames are browsed to check if the behavior is correctly classified. If a misclassification exists, all frames corresponding to the behavior are selected, and the behavior is simultaneously corrected by updating the behavior flag in the label field. 
3. The corrected behavior labels/classes are then added to the previous behavior classifier training dataset to expand the behavior classification training dataset further and quickly retrain the classifier for improved classification moving forward. These cyclic steps in S2 and S3 are performed until the behavior inference classifier achieves a sufficient level of accuracy. \
Note: ‘sufficient’ accuracy is dependent on the study system.
4. The animal detector can also be retrained based on the requirement of the study. You can do this by adding new annotated data from S3 to S1 and following the steps mentioned in S1.

**At this point, you should be ready to rapidly generate large amounts of annoated data which can be used for training other machine learning models or directly used for behavioral analysis for your study**

## Adding a new behavior
1. Add the new behavior to the `--labelflags` array in S3.1. when you load Smarter-labelme.
2. Follow steps in S2 and retrain the behavior classifier to now detect the new behavior along with the previously trained behaviors.