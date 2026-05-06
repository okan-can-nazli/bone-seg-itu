import segmentation_models_pytorch as smp
import torch.nn as nn


def get_model():
    model = smp.Unet(
    encoder_name="resnet34", # feature extractor backbone (goes down)
    encoder_weights="imagenet",   # pretrained on ImageNet, transfer learning
    in_channels=3, # rgb channels on tensor
    classes=1, # output has 1 channel
    activation=None, # BCEWithLogitsLoss applies sigmoid internally
)
    return model