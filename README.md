# CNN-Based Real-Time Audio Alarm Classification System

This source describes a CNN-based real-time audio collection and alarm classification system.

**Items:** CNN training code, trained network weights, and real-time audio inference GUI

Developed for the 2026 Open Source Programming Team Project.

---

## 1. Project Overview

This project continuously collects surrounding sound through a microphone and classifies it into three alarm categories using a trained CNN model.

The supported classes are:

1. `0_indoor_alarms` — Indoor alarm sounds
2. `1_outdoor_warnings` — Outdoor warning sounds
3. `2_emergency_alarms` — Emergency alarm sounds

The real-time module automatically starts microphone input when the program window opens. Audio is collected at 22,050 Hz in mono format and analyzed in 3-second windows with a 1.5-second overlap.

---

## 2. Files

1. `alarm_module_improved.py`

   * Real-time microphone input and GUI inference program
   * Performs noise reduction, silence trimming, peak normalization, Mel Spectrogram conversion, CNN prediction, and result display

2. `my_audio_weights.weights.h5`

   * Trained CNN weight file
   * Must be placed in the same directory as `alarm_module_improved.py`

3. `CNN(6_18).ipynb`

   * Google Colab training code
   * Performs preprocessing, six audio augmentation methods, CNN training, evaluation, and weight-file export

> If the downloaded weight file is named `my_audio_weights.weights(1).h5`, rename it to `my_audio_weights.weights.h5`.

---

## 3. Main Features

* Real-time microphone input using `sounddevice`
* Queue-based audio buffer and multithreaded processing
* 0.1-second audio block collection
* 3-second analysis window with 1.5-second overlap
* Noise reduction, silence trimming, and peak normalization
* 64×64 Mel Spectrogram feature extraction
* CNN-based classification using TensorFlow/Keras
* Prediction smoothing using the latest three inference results
* Low-volume and low-confidence classification rejection
* Tkinter GUI displaying status, predicted class, confidence, RMS level, and class probabilities

---

## 4. CNN Model Structure

The CNN model consists of:

* Input: `64 × 64 × 1`
* Rescaling layer
* Conv2D: 32 filters
* MaxPooling2D
* Conv2D: 64 filters
* MaxPooling2D
* Conv2D: 128 filters
* MaxPooling2D
* Flatten
* Dense: 128 units
* Dropout: 0.4
* Dense: 3 units with Softmax activation

---

## 5. Training Process

The model is trained in Google Colab using WAV files stored in Google Drive.

Recommended dataset structure:

```text
MyDrive/
└── 오디오파일wav/
    ├── 0_indoor_alarms/
    │   └── *.wav
    ├── 1_outdoor_warnings/
    │   └── *.wav
    └── 2_emergency_alarms/
        └── *.wav
```

The training code applies the following preprocessing steps:

1. Noise reduction
2. Silence trimming
3. Peak normalization
4. Mel Spectrogram conversion
5. Resize to `64 × 64`

The training set additionally uses six augmentation methods:

* Time shift
* Pitch shift
* Time stretch
* Noise addition
* Volume change
* Time masking

The trained model and weights are saved as:

```text
my_audio_model.keras
my_audio_weights.weights.h5
```

---

## 6. Environment

Recommended environment:

* Windows 10 or 11
* Anaconda or Miniconda
* Python 3.10 or 3.11
* Microphone input device

Required Python packages:

```text
tensorflow
numpy
sounddevice
librosa
noisereduce
opencv-python
soundfile
scipy
```

---

## 7. Installation

Create and activate an Anaconda virtual environment:

```bash
conda create -n audio_alarm python=3.11
conda activate audio_alarm
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow numpy sounddevice librosa noisereduce opencv-python soundfile scipy
```

---

## 8. How to Run

Place the following files in the same folder:

```text
project_folder/
├── alarm_module_improved.py
└── my_audio_weights.weights.h5
```

Run the program:

```bash
python alarm_module_improved.py
```

The GUI window opens automatically, loads the CNN weights, and starts microphone input.

The program continuously displays:

* Current audio-buffer status
* Predicted sound category
* Confidence score
* Input RMS level
* Probability of each class

Close the GUI window to stop microphone input and terminate the program.

---

## 9. Notes

* Allow microphone access in Windows privacy settings.
* The weight file name must be exactly `my_audio_weights.weights.h5`.
* The real-time model structure must remain identical to the structure used during training.
* Sounds not included in the three training classes may still be classified as one of the classes.
* Low-confidence results are displayed as classification pending.
* Classification performance depends on the quantity, diversity, and recording environment of the training data.

---

## 10. Technologies

* Python
* TensorFlow / Keras
* librosa
* OpenCV
* sounddevice
* noisereduce
* NumPy
* Tkinter
* Google Colab
* Anaconda

---

## 11. License

*This project is licnsed under the MIT License

MIT License

Copyright (c) 2026 Open Source Programming Team Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the commercial software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
