pip install nvidia-pyindex
pip install -r requirements.txt
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip install -U pip && pip install -r requirements.txt
pip install -v -e .
cd ..
python setup.py