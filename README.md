## Requirements
- PyTorch (1.0+)
- python 3

## Installation (development)
### Install pycocotools
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
```

### Get data
Download data from the shared drive and store inside `/data` directory.

### Install rest
Install in editable mode with `python3 -m pip install -e .` so that modifications
in the repository are automatically synced with the installed library
