# encoding: utf-8
"""
Start your Flask server and run this from the terminal as

python simple_request.py --file="data/raw/catsu-cat.png"
"""
import requests
import argparse

# Initialize the PyTorch REST API endpoint URL.
PyTorch_REST_API_URL = "http://127.0.0.1:5000/predict"


def predict_result(image_path):
    # Initialize image path
    image = open(image_path, "rb").read()
    payload = {"image": image}

    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL, files=payload).json()
    # print(r)

    # # Ensure the request was successful.
    if r["success"]:
        # Loop over the predictions and display them.
        print(f"The predicted class is {r['prediction']}")
    # # Otherwise, the request failed.
    else:
        print("Request failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification demo")
    parser.add_argument("--file", type=str, help="test image file")

    args = parser.parse_args()
    predict_result(args.file)
