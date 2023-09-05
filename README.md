# ComputerVisionToSupportNeuroscience

🗓 Spring 2020

## Project  Description

Mouse Behavioural studies are often conducted in the field of neuroscience. Typically the mouse are monitored for changes in sensory-motor function, social interactions, anxiety/ depressive like behavior and other cognitive functions. Traditionally this is done manually by someone watching the recordings and tracking how the mouse moves and interact with each other. This project uses computer vision to track interactions between mouse and their trajectory.

## Demonstration

### Mouse Trajectory
![trajectory](<HTML Report/trajectory.png>)

### Mouse Interactions
Striped Tail Mouse initiates approach | Plain Tail Mouse initiates approach | 
--------------- | --------------- | 
![m1](<HTML Report/mouse1.gif>) | ![m2](<HTML Report/mouse2.gif>)

## Technical Details


1. Cleaning data
![data](<HTML Report/data.png>)
- 30 minute footage of two mice  in a “shoebox” set up, are cut down into 6 footages each of 5 minutes duration.
- Frames to be annotated are selected from the videos in a randomly and temporally uniformly distributed way by clustering on visual appearance (K-means).

2. Annotating frames
- Selected frames are manually annotated by placing trackers on snout, ears and tail base
- Usage of pre-trained Resnet 50 networks in transfer learning to annotate frames in between manually annotated frames.

3. Plot Trajectories received from model

4. Back Tracking Algorithm with coordinates to analyze interaction
![backtracking](<HTML Report/backtracking.png>)
- Detect interaction (an interaction as the event when the two mice got into close proximity of each other. (within ear to ear distance of each other)
- Take the two closest points between the mice and calculated the middle point of these two points to constructed a "collision region". 
- Backtrack through trajectory, the mouse to last enter collision region is the one who initiated the interaction





## Tools
<a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a>
<a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a>
<a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a>
DeepLabCut
