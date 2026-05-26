# Deepfake Detector - Simple Version
import cv2
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image

print("🔄 Starting Deepfake Detector...")
device = torch.device('cpu')

# Face detector
mtcnn = MTCNN(keep_all=True, device=device)

# Simple EfficientNet model
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("✅ Ready!")
print("🎥 Opening webcam...")
print("👉 Press Q to quit")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            face = rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_tensor = transform(Image.fromarray(face)).unsqueeze(0)

            with torch.no_grad():
                out = model(face_tensor)
                prob = torch.softmax(out, dim=1)
                real = prob[0][1].item()
                fake = prob[0][0].item()

            if real > 0.5:
                label = f"REAL {real*100:.1f}%"
                color = (0, 255, 0)
            else:
                label = f"FAKE {fake*100:.1f}%"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Deepfake Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Done!")