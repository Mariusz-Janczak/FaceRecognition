# FaceRecognition

Face detection and comparison based on [DeepFace](https://github.com/serengil/deepface)
It supports various models (ArcFace, VGG-Face, Facenet) and various face detectors (RetinaFace, MTCNN, dlib, OpenCV).

## Installation

```bash
git clone https://github.com/Mariusz-Janczak/FaceRecognition.git
cd FaceRecognition
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

```bash
python -m face_recognition.compare image1.jpg image2.jpg --model ArcFace --detector retinaface
```

### Options

| Option | Default | Description |
| --- | --- | --- |
| --model | ArcFace | Face recognition model (VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace, GhostFaceNet, Buffalo_L) |
| --detector | retinaface | Face detector (retinaface, mtcnn, opencv, dlib) |
| --threshold | None | Custom threshold. If not set it is used default for model. |
