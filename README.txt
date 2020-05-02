CS 585 Final Project

Project Title: Computer Vision to Support Neuroscience
Group Members: Vivian Gunawan, Yaan Tzi Kan, Beatrice Tanaga

dependencies:
	pandas
	pathlib
	numpy
	matplotlib
	tensorflow 1(important)
	wxtools
	deeplabcut (installation instructions:https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md)
	opencv
	math
	scipy


Instructions:
1. Download this directory to your local machine
2. On your terminal/command line interface run "python PostTracking.py"
	The Program calls extract frames which will extract key frames where collisions happen.
	uncomment the elif block in extract frames to get 3 seconds of frames before the key frames.


Results:
1. collision[i].csv can be found in results/csv directory 
	it shows the results for frame and approaches	
	write details on how to interpret this data
2. Frames can be found in results/frames directory
3. Pre-collision Frames can be found in results/preframe directory

