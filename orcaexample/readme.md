# Orca PyTorch video object detection example on tiny-kinectic-400 dataset

We demonstrate how to easily run synchronous distributed Pytorch training using Pytorch Estimator of Project Orca in Bigdl. This is an example using [SlowFast](https://github.com/facebookresearch/SlowFast/tree/main/slowfast) to train on [ting-kinetic-400](https://github.com/Tramac/tiny-kinetics-400), a collection of video clips that cover 400 human action classes. We provide `bigdl` distributed PyTorch training backends for this example. You can run with either backend as you wish.

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environment, especially if you want to run on a yarn cluster (yarn-client mode only).
```
conda create -n bigdl python=3.7  # "bigdl" is conda environment name, you can use any name you like.
conda activate bigdl

# install bigdl-orca
pip install bigdl-orca[ray]
```
Install slowfast **Requirements** refer to their offical document [here](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md#requirements). You don't need to build or install slowfast itself for now.
**Note**:
1. You do **not** need to install pytorch from source like [they suggest](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md#pytorch).
2. PIL is renamed to [Pillow](https://pypi.org/project/Pillow/), but their document hasn't been updated.
3. Their **torchvision** video decode backend are disabled for now, using pyav backend.

Clone **our** slowfast repo:
```
git clone https://github.com/facebookresearch/slowfast
```

## Prepare Dataset
This example only shows setup pipeline, so instead of using Kinetics(135GB), there is a public dataset much smaller here named tiny-kinetics-400(400MB), here goes [links](https://github.com/Tramac/tiny-kinetics-400), and we also provide a even smaller dataset called [tiny-kinetics-4](https://github.com/leonardozcm/SlowFast/releases/tag/tinykinectic400)(6MB) with only 4 classes. You can choose the one you like:
1. tiny-kinetics-4:
```
cd slowfast/orcaexample

wget https://github.com/leonardozcm/SlowFast/releases/download/tinykinectic400/tiny-kinetics-400.zip
```


## Install slowfast
Please make sure your slowfast works--After having the above dependencies, run:
```
git clone https://github.com/facebookresearch/slowfast
cd SlowFast
python setup.py build develop
```
## Run example
You can run this example on local mode (default) and yarn-client mode.

- Run with Spark Local mode:
```bash
python kinectic.py --cluster_mode local
```

- Run with Yarn-Client mode:
```bash
python kinectic.py --cluster_mode yarn
```

You can run this example with bigdl backend (default), ray backend, or spark backend. 

- Run with bigdl backend:
```bash
python kinectic.py --backend bigdl
```

- Run with ray backend:
```bash
python kinectic.py --backend ray
```

- Run with spark backend:
```bash
python kinectic.py --backend spark
```

**Options**
* `--cluster_mode` The mode of spark cluster. Either "local" or "yarn". Default is "local".
* `--backend` The backend of PyTorch Estimator. Either "bigdl", "ray" or "spark. Default is "bigdl".

## Results

**For "bigdl" backend**

**For "ray" and "spark" backend**
