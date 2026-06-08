# Image Classification Template (TensorFlow + Gradio)

Train your own image classifier and launch it in a web browser.

## Features

* Train on your own images
* Works with any image classes
* TensorFlow + VGG16 transfer learning
* Gradio web interface
* Confidence scores
* Confusion matrix
* Classification report
* Automatic train/test split

## Installation

Install dependencies:

```bash
run :
pip install -r requirements.txt
```

## Dataset Structure

Example:

```text
dataset/
├── cats/
│   ├── image1.jpg
│   └── image2.jpg
│
├── dogs/
│   ├── image1.jpg
│   └── image2.jpg
```

You can use any class names.

## Training

Run:

```bash
python train.py
```

You may either:

* Press Enter to use the included `dataset` folder
* Provide the path to your own dataset

When running:

python train.py

you will see:

Dataset folder path (press Enter for './dataset'):

You can:

Press Enter to use:

dataset/

or provide your own path:

Windows:

C:\Users\John\Downloads\CatsVsDogs

Linux / macOS:

/home/john/datasets/cats_vs_dogs

The dataset folder should contain class folders exactly as shown in the examples above.

Example:

```text
C:\Datasets\CatsVsDogs
```

The trained model is saved automatically:

```text
saved_model/model.h5
```

## Launch the Interface

Run:

```bash
python GUI.py
```

Upload an image and receive:

* Predicted class
* Confidence score
* Probability distribution

## Notes

* At least two classes are required.
* More images generally improve performance.
* The first training run may take several minutes depending on hardware.

## License

Personal and commercial use.
