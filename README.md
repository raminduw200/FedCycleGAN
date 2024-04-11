# FedCycleGAN: Enhancing Mobile Photography through Federated Learning-Enabled CycleGAN

Mobile image photography is continuously emerging as an area of interest, yet achieving professional-level camera quality remains a challenge due to hardware limitations. Different
smartphone manufacturers follow different methods, resulting in varying color schemes for the output image.


This thesis proposes a novel model that trains in a distributed manner while preserving the privacy of user-captured images. The proposed solution addresses the issue of collecting paired images in CNN networks by deploying a CycleGAN model. The proposed CycleGAN model is trained on user-captured RAW images with pre-loaded target domain images (professional camera-captured images). The training process is distributed using federated learning. The federated learning concept not only allows training in a distributed manner but also extends
the architecture for continuous training. Both GANs of the CycleGAN are synced with the central federated server and utilize the FedAvg algorithm to aggregate all two GANs. The central server, hosted in the cloud, trains clients in different geo-locations. As clients continuously train their CycleGAN model, aggregation in the central server speeds up the learning process. This achieves uniform image processing across multiple smartphones, even though their hardware specifications may differ. This method can be adapted for third-party applications that use smartphone cameras.

The proposed novel FedCycleGAN architecture for enhancing image colors processes an image in under 0.005 seconds. Additionally, it surpasses most state-of-the-art models in terms of PSNR and SSIM values. The proposed method is capable of rendering high-resolution images with 7.4GB VGA RAM, while benchmark models ran out of VGA memory. Furthermore, human subjective analysis validates the quality of the generated outputs compared to the ground truth images

## Training
For training, define the dataroot and the name for saving the model. Wandb has configured, if not using it, can drop the `--use_wandb --wandb_project_name FedCycleGAN-2` parameters.

```
python main.py --dataroot /path/to/train/images --name RTX-3080-epoch10-fl10 --model cycle_gan --client_count 2 --n_epochs 5 --n_epochs_decay 5 --batch_size 1 --save_epoch_freq 1 --fl_rounds 10 --use_wandb --wandb_project_name FedCycleGAN-2
```

If you wish to continue the training process from the last saving point, you can add flag `--continue_train`.
```
python main.py --dataroot /path/to/train/images --name RTX-3080 --model cycle_gan --client_count 2 --n_epochs 3 --n_epochs_decay 3 --batch_size 1 --save_epoch_freq 1 --fl_rounds 5 --load_iter 6 --continue_train
```

## Testing
Please specify the dataroot path for test images.
```
python test.py --dataroot /path/to/train/images --name RTX-3080-FL-FINAL-continue-fl10 --model cycle_gan --no_dropout --eval --num_test 1204 --preprocess none
```


## Backend API
The dataroot parameter here does not use since it is hardcoded in the `backend_api.py` file. Running the testing model in `test` specify to load a single generator to enhance images.
```
python backend_api.py --dataroot / --name RTX-3080-FL-FINAL-continue-fl10 --model test --no_dropout --eval
```