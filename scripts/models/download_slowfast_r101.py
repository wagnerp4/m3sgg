import torch
from pytorchvideo.models.hub import slowfast_r101


def main():
    print("Loading SlowFast R101 8x8 model using PyTorchVideo ...")
    model = slowfast_r101(pretrained=True)  # This downloads and loads the correct model
    print("Model loaded successfully!")


if __name__ == "__main__":
    main()
