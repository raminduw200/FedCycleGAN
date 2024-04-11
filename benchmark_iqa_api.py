import cv2
import os
from PIL import Image
import pyiqa
import torch

# ['ahiq', 'brisque', 'ckdn', 'clipiqa', 'clipiqa+', 'clipiqa+_rn50_512', 'clipiqa+_vitL14_512', 'clipscore', 'cnniqa', 'cw_ssim', 'dbcnn', 'dists', 'entropy', 'fid', 'fsim', 'gmsd', 'hyperiqa', 'ilniqe', 'laion_aes', 'liqe', 'liqe_mix', 'lpips', 'lpips-vgg', 'mad', 'maniqa', 'maniqa-kadid', 'maniqa-koniq', 'maniqa-pipal', 'ms_ssim', 'musiq', 'musiq-ava', 'musiq-koniq', 'musiq-paq2piq', 'musiq-spaq', 'nima', 'nima-koniq', 'nima-spaq', 'nima-vgg16-ava', 'niqe', 'nlpd', 'nrqm', 'paq2piq', 'pi', 'pieapp', 'psnr', 'psnry', 'ssim', 'ssimc', 'stlpips', 'stlpips-vgg', 'topiq_fr', 'topiq_fr-pipal', 'topiq_iaa', 'topiq_iaa_res50', 'topiq_nr', 'topiq_nr-face', 'topiq_nr-flive', 'topiq_nr-spaq', 'tres', 'tres-flive', 'tres-koniq', 'uranker', 'vif', 'vsi']
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
psnr_ = pyiqa.create_metric('psnr', device=device)
print("PSNR: ", psnr_.lower_better)
ssim_ = pyiqa.create_metric('ssim', device=device)
print("SSIM: ", ssim_.lower_better)
niqe_ = pyiqa.create_metric('niqe', device=device)
print("NIQE: ", niqe_.lower_better)
brisque_ = pyiqa.create_metric('brisque', device=device)
print("BRISQUE: ", brisque_.lower_better)
fid_ = pyiqa.create_metric('fid', device=device)
print("FID: ", fid_.lower_better)


def print_metrics(num_images, metrics):
    avg_psnr = sum(metrics["PSNR"][:num_images]) / num_images
    avg_ssim = sum(metrics["SSIM"][:num_images]) / num_images
    avg_niqe = sum(metrics["NIQE"][:num_images]) / num_images
    avg_brisque = sum(metrics["BRISQUE"][:num_images]) / num_images

    print(f"------------------------ Images: {num_images} ----------------------------")
    print(f"Average PSNR: {avg_psnr}")
    print(f"Average SSIM: {avg_ssim}")
    print(f"Average NIQE: {avg_niqe}")
    print(f"Average BRISQUE: {avg_brisque}")


def calculate_metrics(folder1, folder2, csv_name="image_metrics.csv"):
    # List all image files in both folders
    images1 = sorted(os.listdir(folder1))
    images2 = sorted(os.listdir(folder2))
    num_images = 0
    metrics = {"PSNR": [], "SSIM": [], "NIQE": [], "BRISQUE": []}
    total_psnr, total_ssim, total_niqe, total_brisque, total_fid = 0, 0, 0, 0, 0

    # print dimensions of first image images1 and images2
    print("Image 1 dimensions: ", cv2.imread(os.path.join(folder1, images1[0])).shape)
    print("Image 2 dimensions: ", cv2.imread(os.path.join(folder2, images2[0])).shape)

    # Iterate through each pair of images with the same name
    for i in range(len(images1)):
        # filter only images ends with _fake_B.png
        if ("RTX-3080-FL-FINAL" in folder1 or "enlightening" in folder1 or "FedCycleGAN" in folder1) and not images1[i].endswith("_fake_B.png"):
            continue

        img_path1 = os.path.join(folder1, images1[i])
        if "RTX-3080-FL-FINAL" in folder1 or "enlightening" in folder1 or "FedCycleGAN" in folder1:
            img2_name = images1[i].split(".")[0].split("_")[0] + ".jpg"
        else:
            img2_name = images1[i].split(".")[0] + ".jpg"
        # img_path2 = os.path.join(folder2, images1[i].split(".")[0] + ".jpg")
        img_path2 = os.path.join(folder2, img2_name)

        # Read images
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        # Reshape images to the same size
        if "RTX-3080-FL-FINAL" in folder1 or "DPED" in folder1 or "enlightening" in folder1 or "FedCycleGAN" in folder1:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        # Calculate PSNR
        psnr_value = psnr_(img1, img2)
        ssim_value = ssim_(img1, img2)
        niqe_value = niqe_(img1)
        brisque_value = brisque_(img1)
        # fid_value = fid_(img1, img2)
        # print(f"Image: {images1[i]} --> {img2_name},  PSNR: {psnr_value}, SSIM: {ssim_value}, NIQE: {niqe_value}, BRISQUE: {brisque_value}")

        metrics["PSNR"].append(psnr_value)
        metrics["SSIM"].append(ssim_value)
        metrics["NIQE"].append(niqe_value)
        metrics["BRISQUE"].append(brisque_value)
        total_psnr += psnr_value
        total_ssim += ssim_value
        total_niqe += niqe_value
        total_brisque += brisque_value

        # Write results to CSV
        # writer.writerow({"Image": images1[i], "PSNR": psnr_value, "SSIM": ssim_value})
        num_images += 1

    # sort the metrics by highest PSNR value
    metrics["PSNR"] = sorted(metrics["PSNR"], reverse=True)
    metrics["SSIM"] = sorted(metrics["SSIM"], reverse=True)
    metrics["NIQE"] = sorted(metrics["NIQE"], reverse=True)
    metrics["BRISQUE"] = sorted(metrics["BRISQUE"], reverse=True)

    print("folder1: ", folder1)
    print("folder2: ", folder2)

    print_metrics(int(num_images*0.25), metrics)
    print_metrics(int(num_images*0.5), metrics)
    print_metrics(int(num_images*0.75), metrics)
    print_metrics(int(num_images), metrics)


# FedCycleGAN ResNET 9 -1176
# folder1 = "/home/ramindu/Documents/FedCycleGAN/results/RTX-3080-FL-FINAL-continue-fl10/cropped_1776_iter3/images/"
folder1 = "/home/ramindu/Documents/FedCycleGAN/results/RTX-3080-FL-FINAL-continue-fl10/test_latest/images/"
folder2 = "/media/ramindu/50521afa-8e18-4599-9117-2d9df879a8b5/Dataset/zurich-taw-to-dslr-dataset/test/cropped/test-RAW-Visualized/canon/"

# FedCycleGAN UNET 256 - 1176
# folder1 = "/home/ramindu/Documents/FedCycleGAN/results/FedCycleGAN-2Clients-Unet256-Test/test_latest/images/"
# folder2 = "/media/ramindu/50521afa-8e18-4599-9117-2d9df879a8b5/Dataset/zurich-taw-to-dslr-dataset/test/cropped/test-RAW-Visualized/canon/"

# # CycleGAN-MIE-VGGLoss - 1176
# folder1 = "/home/ramindu/Documents/FedCycleGAN/results/CycleGAN-MIE-VGGLoss/test_latest/images/"
# folder2 = "/media/ramindu/50521afa-8e18-4599-9117-2d9df879a8b5/Dataset/zurich-taw-to-dslr-dataset/test/cropped/test-RAW-Visualized/canon/"

# PyNET V1 - checked
# folder1 = "/media/ramindu/50521afa-8e18-4599-9117-2d9df879a8b5/Projects/PyNET-PyTorch/results/eval-1776/"
# folder2 = "/media/ramindu/50521afa-8e18-4599-9117-2d9df879a8b5/Dataset/zurich-taw-to-dslr-dataset/test/cropped/test-RAW-Visualized/canon/"

# EnlighteningGAN
# folder1 = "/media/ramindu/50521afa-8e18-4599-9117-2d9df879a8b5/Projects/EnlightenGAN/ablation/enlightening/test_200/images/"
# folder2 = "/media/ramindu/50521afa-8e18-4599-9117-2d9df879a8b5/Dataset/zurich-taw-to-dslr-dataset/test/cropped/test-RAW-Visualized/canon/"

# DPED 1776 - checked
# folder1 = "/media/ramindu/50521afa-8e18-4599-9117-2d9df879a8b5/Projects/DPED/visual_results/eval-1776/"
# folder2 = "/media/ramindu/50521afa-8e18-4599-9117-2d9df879a8b5/Dataset/zurich-taw-to-dslr-dataset/test/cropped/test-RAW-Visualized/canon/"

# csv_name = "benchmarks/DPED-1176.csv"
calculate_metrics(folder1, folder2)
