## Requirements
- PyTorch (1.0+)
- python 3

## Installation (development)
### Install pycocotools
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
python setup.py build_ext install
```

### Get data
Download data from the shared drive and store inside `/data` directory.
The resulting folder structure should look something like

```
.
├── data
│   ├── PennFudanPed
│   │   ├── added-object-list.txt
│   │   ├── Annotation
│   │   ├── PedMasks
│   │   ├── PNGImages
│   │   └── readme.txt
│   └── synthetic-val
│       ├── cup-with-waves-val
│       ├── flower-bath-bomb-val
│       ├── heart-bath-bomb-val
│       ├── square-plastic-bottle-val
│       └── stemless-plastic-champagne-glass-val
├── README.md
├── sem_to_ins
│   ├── cfg.py
│   ├── dataset.py
│   ├── detection_reference
│   │   ├── coco_eval.py
│   │   ├── coco_utils.py
│   │   ├── engine.py
│   │   ├── __init__.py
│   │   ├── transforms.py
│   │   └── utils.py
│   └── __init__.py
└── setup.py
```

### Install rest
Install in editable mode with `python3 -m pip install -e .` so that modifications
in the repository are automatically synced with the installed library
