## Neural Relational Inference with Efficient Message Passing Mechanisms
This repository contains the source code for the paper *Neural Relational Inference with Efficient Message Passing Mechanisms* accepted by AAAI 2021. [arXiv](https://arxiv.org/pdf/2101.09486)

## Requirements
- Ubuntu 16.04
- python 3.6
- pytorch >= 1.2.0
- numpy >= 1.14.5
- scipy >= 1.1.0
- torch-geometric >= 1.3.2
- CUDA 10.0

Please follow the instructions in the [official site](https://github.com/rusty1s/pytorch_geometric) to successfully install torch-geometric.

## Overview of Models
This repository implements a series of encoders and decoders to allow flexible combinations.

- Encoders
  - GNNENC: original encoder (MLP & CNN) of NRI.
  - RNNENC: encoder with the relation interaction mechanism implemented by RNNs.
    - `option='node'`: use only the intra-edge relation interaction mechanism.
    - `option='edge'`: use only the inter-edge relation interaction mechanism.
  - AttENC: encoder with the relation interaction mechanism implemented by self-attention.
- Decoders
  - GNNDEC: original decoder (MLP) of NRI.
  - RNNDEC: decoder with the spatio-temporal message passing mechanism implemented by RNNs.
    - `option='node'`: use only the node-level spatio-temporal operation.
    - `option='edge'`: use only the edge-level spatio-temporal operation.
  - AttDEC: decoder with the spatio-temporal message passing mechanism implemented by RNNs and the temporal attention mechanism.

#### Implementation tricks for permutation equivariant operations
The most tricky part is the implementation of the permutation equivariant operations in the RNN based relation interaction mechanism. The key is to **keep track of the indices**. Here is an examples.
```python
import numpy as np

# original indices
original = np.arange(10)
# permuted indices
index = np.random.permutation(original)
# index mapping to recover the indices
inv_index = index.copy()
inv_index[index] = original
# recovered indices
recover = index[inv_index]
# check if correctly reovered
print((original == recover).all())
```

## Data generation
Generate a 5-object Springs dataset. The parameter `interval` means the down-samping factor. Replace `spring` with `charged` in the following code to generate a 5-object Charged dataset.
```bash
python generate.py --dyn spring --size 5 --interval 100
```
Generate a 5-object Kuramoto dataset.
```bash
python generate.py --dyn kuramoto --size 5 --interval 10
```
Typically, the size of a 5-object dataset is around `619M`, and that of a 10-object dataset is around `1.3G`.

## Run experiments
Reproduce the results of NRI-MPM in the 5-obejct Springs, Charged and Kuramoto dataset, respectively.

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --dyn spring --reduce mlp --dim 4 --size 5 --enc RNNENC --dec RNNDEC --reg 1e2 --scheme both
```

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --dyn charged --reduce cnn --dim 4 --size 5 --enc RNNENC --dec RNNDEC --reg 1e2 --scheme both
```

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --dyn kuramoto --reduce cnn --dim 3 --skip --size 5 --enc RNNENC --dec RNNDEC --reg 1 --scheme both
```

Since reproducing the results in the 10-object datasets requires more memory, you may specify multiple GPUs, e.g.,

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py --dyn spring --reduce mlp --dim 4 --size 10 --enc RNNENC --dec RNNDEC --reg 1e2 --scheme both
```

Train the encoder in a supervised manner.
```bash
CUDA_VISIBLE_DEVICES=0 python run.py --dyn spring --reduce mlp --dim 4 --size 5 --enc RNNENC --scheme enc
```

Train the decoder given the ground truth interacting relations.
```bash
CUDA_VISIBLE_DEVICES=0 python run.py --dyn spring --reduce mlp --dim 4 --size 5 --dec RNNDEC --scheme dec
```

## Citation
If you find this repository useful in your research, please consider citing the following paper:
```
@inproceedings{chen2021nrimpm,
	title={Neural Relational Inference with Efficient Message Passing Mechanisms},
	author={Chen, Siyuan and Wang, Jiahai and Li, Guoqing},
	booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
	pages={},
	year={2021}
}
```
