## Installation Guide 
#### Clone Repository
1. git clone git@github.com:YiweiC1W/pedestrians_detection.git # or git clone  https://github.com/YiweiC1W/pedestrians_detection.git
2. cd pedestrians_detection

#### Create virtual environment
3. conda create --name 9517gp python==3.8.13
4. conda activate 9517gp

#### Install dependencies according to your system
5. sh install_linux.sh # For Linux 
6. install_win64.bat # For Windows



## Run

### Arguments

please edit IMAGE_FOLDER_PATH in task1.py

or you can use args '--path' to choose your image folder

'--conf' to choose your confidence threshold

'--device' to choose your device 'cpu' or 'gpu'

'--video' bool(true or false), if you want to save video, default is true

'--picture' bool(true or false), if you want to save picture, default is true



### run example
python task1.py


## 如何更新代码

如果使用pycharm:  pycharm -> git 菜单 -> update project


如果使用git 命令行: git pull




如果更新过程中有代码冲突， 尽量使用手动 merge 避免被覆盖。
