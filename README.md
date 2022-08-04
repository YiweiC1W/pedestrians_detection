## Installation Guide 
We strongly recommend you to run the following commands from the system shell or terminal.

**DO NOT RUN these commands from any IDE terminal. There are some strange issues especially from PyCharm Terminal**



### 1. Clone Repository (if you clone from github, use this step)
1.1 Clone the repository from github

     git clone git@github.com:YiweiC1W/pedestrians_detection.git

     git clone  https://github.com/YiweiC1W/pedestrians_detection.git

1.2 cd to the directory

     cd pedestrians_detection


### 1. Unzip Folder (if you downloaded the zip file, use this step)

Unzip the folder

cd into the project root folder (where the README.md is)

### 2. Create virtual environment
2.1 Create virtual environment 
   
     conda create --name 9517gp python==3.8.13

2.2 Activate the virtual environment

     conda activate 9517gp

### 3.Run this All-in-One script to install dependencies, YOLOX, Download datasets and pre-trained weights (choose one according to your system)

     sh install_linux.sh # For Linux (Ubuntu 20.04 is recommended) (maybe works on macOS too)

     install_win64.bat # For Windows (not tested)

If you encounter any installation issues, please contact me at yiwei.chen2@student.unsw.edu.au


## Run

### Arguments

please edit IMAGE_FOLDER_PATH in task1.py

or you can use args '--path' to choose your image folder

'--task' to choose your task eg: '--task 1' or '--task 2'

'--conf' to choose your confidence threshold

'--device' to choose your device 'cpu' or 'gpu'

'--video' bool(true or false), if you want to save video

'--picture' bool(true or false), if you want to save picture



### run example
 python main.py --task 1 --device cpu


## Code Reference

[1] https://github.com/nwojke/deep_sort

[2] https://github.com/Megvii-BaseDetection/YOLOX