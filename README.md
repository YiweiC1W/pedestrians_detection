## Installation Guide 
1. git clone git@github.com:YiweiC1W/pedestrians_detection.git # or git clone  https://github.com/YiweiC1W/pedestrians_detection.git
2. cd pedestrians_detection
3. conda create --name 9517gp python==3.8.13
4. conda activate 9517gp
5. pip install tensorrt
6. pip install nvidia-pyindex
7. pip install -r requirements.txt
8. git clone https://github.com/Megvii-BaseDetection/YOLOX.git
9. cd YOLOX
10. pip install -r requirements.txt # 这一步可能不需要, 但跑一下也没事
11. pip install -v -e .   # or  python3 setup.py develop

!注意上一步最后有一个 '.' 符号  

## Run

### Arguments

please edit IMAGE_FOLDER_PATH in task1.py

or you can use args '--path' to choose your image folder

'--conf' to choose your confidence threshold

'--device' to choose your device 'cpu' or 'gpu'

'--video' bool(true or false), if you want to save video, default is true

'--picture' bool(true or false), if you want to save picture, default is true



### run example
python3 task1.py


