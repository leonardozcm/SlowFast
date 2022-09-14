# Orca PyTorch video object detection example on tiny-kinectic-400 dataset

We demonstrate how to easily run synchronous distributed Pytorch training using Pytorch Estimator of Project Orca in Bigdl. This is an example using the efficient sub-pixel convolution layer to train on [BSDS3000 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), using crops from the 200 training images, and evaluating on crops of the 100 test images. See [here](https://github.com/pytorch/examples/tree/master/super_resolution) for the original single-node version of this example provided by Pytorch. We provide three distributed PyTorch training backends for this example, namely "bigdl", "ray" and "spark". You can run with either backend as you wish.

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environment, especially if you want to run on a yarn cluster (yarn-client mode only).
```
conda create -n bigdl python=3.7  # "bigdl" is conda environment name, you can use any name you like.
conda activate bigdl

...
```

## Prepare Dataset

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
