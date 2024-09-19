import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

import torch
from torch import nn, optim, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import PIL
from PIL import Image
import json
from collections import OrderedDict
import argparse


# Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    pre_process = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pil_image_model = pre_process(pil_image)
    return pil_image_model


# Class prediction
def predict(model, device="gpu", top_k=1, image_path="flowers/test/69/image_05959.jpg",
            category_names='cat_to_name.json'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # DONE: Implement the code to predict the class from an image file
    # moving to gpu
    if torch.cuda.is_available() and device == "gpu":
        model.to('cuda')

    # processing image
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    # getting output
    with torch.no_grad():
        output = model.forward(img_torch.cuda()) if device == "gpu" else model.forward(img_torch)

    # getting result possibilites
    ps = F.softmax(output.data, dim=1)

    # Find the top top_k probabilities and classes
    top_p, top_class = ps.topk(top_k, dim=1)

    # Convert to lists
    top_p = top_p.tolist()[0]
    top_class = top_class.tolist()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in
                    model.class_to_idx.items()}
    top_class = [idx_to_class[label] for label in top_class]

    return top_p[0], top_class[0]


