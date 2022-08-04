pip3 install nvidia-pyindex
pip3 install -r 'requirements.txt'
python3 -m pip install -U scikit-image
git clone https://github.com/YiweiC1W/9517dataset.git
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -U pip && pip3 install -r 'requirements.txt'
pip3 install -v -e .
cd ..
python3 setup.py