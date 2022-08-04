pip install nvidia-pyindex
pip install -r requirements.txt
pip install imutils
python -m pip install -U scikit-image
git clone https://github.com/YiweiC1W/9517dataset.git
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip install -U pip && pip install -r requirements.txt
pip install -v -e .
cd ..
python setup.py