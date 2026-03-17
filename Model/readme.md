# 🧠 Vehicle Detection Model

This directory contains the machine learning model and label map required for the Edge-Based Adaptive Headlight System. 

To keep the repository lightweight, the compiled model weights are not tracked by version control. You must download them before running the main pipeline.

## Required Files
You need two files in this directory to run the system:
1. `detect.tflite` - The quantised model weights.
2. `labelmap.txt` - The class labels for the detection boxes.

## Model Details
* **Architecture:** MobileNet-SSD (Single Shot MultiBox Detector)
* **Format:** TensorFlow Lite (`.tflite`)
* **Precision:** INT8 Quantised (for Edge deployment)
* **Training Dataset:** COCO (Common Objects in Context)
* **Target Classes:** Car, Motorcycle, Truck, Bus

## Download Instructions

### Option 1: Automated Download (Linux/macOS)
Run the following commands from within this `model/` directory to download the official starter model provided by TensorFlow:

    # Download the compressed model package
    wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip

    # Unzip the contents
    unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip

    # The zip contains multiple files. You only need these two:
    # 1. detect.tflite
    # 2. labelmap.txt

    # Clean up the zip file
    rm coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip


### Option 2: Manual Download
If you are on Windows or prefer to download manually:
1. Download the zip file from [TensorFlow's Hosted Models](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip).
2. Extract the archive.
3. Copy `detect.tflite` and `labelmap.txt` directly into this folder.

## Custom Models
If you wish to use a custom-trained model (e.g., fine-tuned for night-time vehicle signatures or specific headlight glare patterns), place your custom `.tflite` file here and update the `MODEL_PATH` variable in `main.py`.
