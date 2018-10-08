# load model weights from disk
from pathlib import Path

import numpy as np

# load images
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from model_class import TheModelClass

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

model = TheModelClass()


PATH = Path("models") / "cifar-weights.pt"
PATH = PATH.resolve().absolute()
model.load_state_dict(torch.load(str(PATH)))
model.eval()


def prepare_image(image_path):
    imsize = 32
    data_transforms = transforms.Compose(
        [
            transforms.Resize(imsize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image = data_transforms(image).float()
    image_tensor = torch.tensor(image, requires_grad=True)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


# Make a prediction on single image
image_path = Path("data/raw/catsu-cat.png").resolve().absolute()
print(image_path)
image = prepare_image(str(image_path))
outputs = model(image)
_, predicted = torch.max(outputs, 1)
class_prediction = predicted.numpy()[0]
print(classes[class_prediction])
