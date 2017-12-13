# Object Detection with Tensorflow

Learn how to use this at the docs folder.

# Setup

Follow the steps:

## 1 - Install Miniconda:

https://conda.io/miniconda.html

## 2- Clone this repo:

https://intranet.gradiant.org/bitbucket/projects/VA/repos/object-detection-tensorflow/browse

```bash
cd object-detection-tensorflow
```

## 2 - Create conda enviroment:

### a) If you don't have a GPU:

```bash
conda env create -f env.yml
source activate tensorflow
```

### b) If you have a GPU (i.e. gea):

```bash
conda env create -f env-gpu.yml
source activate tensorflow-gpu
```

The following commands assume you have the enviroment **active**.


## 4 - Compile the protobufs:

```bash
protoc object_detection/protos/*.proto --python_out=.
```

## 5 - Install the internal modules:

```bash
pip install -e .

pip install -e slim
```

## 6 - Test the installation 

Running (from object_detection_tensorflow root):

```bash
python train.py --help
```

## 7 - Download the pretrained models:

```bash
bash get_all_pretrained_models.sh
```

## 8 - (Optional) Install Jupyter

If you want to run the notebooks:

```bash
conda install jupyter
```
