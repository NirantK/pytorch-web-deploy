# encoding: utf-8
"""
@author: Nirant Kasliwal
@contact: ml@nirantk.com
"""

import io
import json
from pathlib import Path

import flask
import torch
import torch.nn.functional as F
from flask import jsonify
from PIL import Image
from torch import nn
from torchvision import transforms as T
import os
import sys

import_path = os.getcwd()
sys.path.insert(0, import_path)


from src.models.model_class import TheModelClass

# Initialize our Flask application and the PyTorch model.
app = flask.Flask(__name__)
use_gpu = False
model = None


@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({"status": True, "version": "0.1.0", "api_version": "v1"})


def prepare_image(image_path):
    imsize = (32, 32)
    data_transforms = T.Compose(
        [T.Resize(imsize), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    image = Image.open(image_path).convert("RGB")
    image = data_transforms(image).float()
    image_tensor = torch.tensor(image, requires_grad=True)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


@app.route("/")
@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

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

    if flask.request.method == "GET":
        return flask.render_template("index.html")

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == "POST":
        print(flask.request.files)
        if flask.request.files.get("image"):
            # Read the image in PIL format
            f = flask.request.files["image"]
            f.save(f.filename)  # save file to disk

            # Preprocess the image and prepare it for classification.
            image = prepare_image(f.filename)

            # Classify the input image and then initialize the list of predictions to return to the client.
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            class_prediction = predicted.numpy()[0]
            data["prediction"] = classes[class_prediction]

            # Indicate that the request was a success.
            data["success"] = True
            os.remove(f.filename)

    # Return the data dictionary as a JSON response
    if data["success"]:
        return flask.render_template("index.html", label=data["prediction"])
    else:
        return flask.jsonify(data)


def load_model() -> None:
    # Define model
    global model
    model = TheModelClass()

    # load model weights from disk
    PATH = Path("models") / "cifar-weights.pt"
    PATH = PATH.resolve().absolute()
    model.load_state_dict(torch.load(str(PATH)))
    model.eval()


if __name__ == "__main__":
    print("Loading PyTorch model and Flask starting server.")
    print("Please wait until server has fully started...")
    load_model()
    app.run(debug=True)
