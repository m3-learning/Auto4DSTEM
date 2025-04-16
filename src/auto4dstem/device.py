import os
import torch


def get_device(cuda_device_order="PCI_BUS_ID", cuda_visible_devices="0"):
    os.environ["CUDA_DEVICE_ORDER"] = cuda_device_order
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    return device
