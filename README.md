## Installation Guide 
#### Clone Repository
1. git clone git@github.com:YiweiC1W/pedestrians_detection.git

     or  git clone  https://github.com/YiweiC1W/pedestrians_detection.git

2. cd pedestrians_detection

#### Create virtual environment
3. conda create --name 9517gp python==3.8.13
4. conda activate 9517gp

#### Install dependencies according to your system
5. sh install_linux.sh # For Linux (Ubuntu 20.04 is recommended!) (maybe works on macOS too)
6. install_win64.bat # For Windows (have not tested)



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