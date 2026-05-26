# 🛡️ SentinelAI — Real-time Image Manipulation Detection

An ML-powered system that detects manipulated/fake images in real-time using Deep Learning and Computer Vision.

## 🎯 Project Overview
SentinelAI uses EfficientNet architecture to analyze facial images through a live webcam feed and classify them as **REAL** or **FAKE** with confidence percentage.

## 🔥 Features
- ✅ Real-time webcam face detection
- ✅ REAL / FAKE classification with confidence %
- ✅ Green box (Real) / Red box (Fake) visualization
- ✅ Built with EfficientNet Deep Learning model
- ✅ Face detection using MTCNN

## 🛠️ Tech Stack
- Python 3.12
- PyTorch
- EfficientNet (timm)
- OpenCV
- MTCNN (facenet-pytorch)

## 🚀 How to Run

```bash
git clone https://github.com/Yogavelan2007/SentinelAI.git
cd SentinelAI
python -m venv venv
venv\Scripts\activate
pip install torch torchvision timm facenet-pytorch opencv-python
python deepfake_detector.py
```

## 📊 How it Works
1. Webcam captures live video feed
2. MTCNN detects faces in each frame
3. EfficientNet model analyzes the face
4. REAL / FAKE label shown with confidence %

## 👨‍💻 Author
**YOGAVELAN M D**   
🔗 [GitHub Profile](https://github.com/Yogavelan2007)  
📁 [Project Repository](https://github.com/Yogavelan2007/SentinelAI)
