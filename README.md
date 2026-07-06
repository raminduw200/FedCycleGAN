# FedCycleGAN

This repository implements a federated learning extension of CycleGAN for image-to-image translation, inspired by the research paper "Enhancing Mobile Photography with Federated Learning-Enabled CycleGAN." The project demonstrates how a CycleGAN model can be trained across multiple distributed clients while preserving client data locality and aggregating a shared central model.

## Overview

FedCycleGAN adapts the CycleGAN architecture to a federated training workflow. Each client trains local CycleGAN models on its own image dataset, then sends model parameters to a central server for aggregation. The aggregated server model is redistributed to clients for the next federated round.

This approach is designed for mobile photography enhancement scenarios where training data is distributed across devices and direct data sharing is undesirable.

## Features

- Federated training across multiple clients
- Support for CycleGAN image translation models
- Aggregation of generator weights into a shared server model
- Local client training on private data directories
- Optional standard CycleGAN training for baseline experiments

## Setup

Recommended Python environment:

- Python 3.8
- PyTorch 1.8.1
- torchvision 0.9.1
- Flower 1.7.0
- visdom
- dominate
- wandb

Install dependencies with `pip`:

```bash
pip install -r requirements.txt
```

Or use the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate pytorch-CycleGAN-and-pix2pix
```

## Usage

### Federated Training

Run the federated workflow with:

```bash
python federated_train.py --dataroot /path/to/fed_dataset --name fed_experiment --model cycle_gan --client_num 2 --num_fed_rounds 5
```

Important options:

- `--dataroot` - root directory containing client dataset folders
- `--client_num` - number of federated clients
- `--num_fed_rounds` - number of federated aggregation rounds
- `--model` - should be set to `cycle_gan`

The federated script expects client data directories named like:

```
/path/to/fed_dataset/client_1
/path/to/fed_dataset/client_2
```

Each client folder should contain image data formatted according to the CycleGAN dataset loader.

### Standalone Training

Train a regular CycleGAN model with:

```bash
python train.py --dataroot /path/to/dataset --name baseline_cyclegan --model cycle_gan
```

This script uses the standard CycleGAN training pipeline and can be useful for comparing federated and centralized performance.

### Flower Client

The repository also includes a Flower client implementation in `client_cyclegan.py`. It can connect to a Flower server and participate in federated learning using the Flower ecosystem.

## Dataset Structure

For federated training, organize data as separate client partitions under the main `--dataroot` folder. Each client should have its own dataset directory, for example:

```
/fed_data/client_1
/fed_data/client_2
```

## Notes

- This repository is built on a CycleGAN/Pix2Pix codebase and extends it with federated learning concepts.
- The federated workflow currently aggregates only the generative models and transmits updated weights back to clients.
- Some experimental features such as PSNR-based aggregation and evaluation hooks are present but may be disabled or commented out.

## License

See `LICENSE` for license details.
