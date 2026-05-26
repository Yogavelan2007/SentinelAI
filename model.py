# model.py — Proper Deepfake Model (HuggingFace)
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

def load_model():
    print("🔄 Loading Proper Deepfake Detection model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")

    # This model is trained specifically on deepfake images!
    model_name = "prithivMLmods/Deep-Fake-Detector-Model"

    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    print("✅ Deepfake model loaded!")
    return model, processor, device


def predict(model, processor, device, face_pil):
    inputs = processor(images=face_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    # Get labels
    id2label = model.config.id2label
    real_prob = 0.0
    fake_prob = 0.0

    for idx, prob in enumerate(probs[0]):
        label = id2label[idx].lower()
        if 'real' in label:
            real_prob = prob.item()
        elif 'fake' in label:
            fake_prob = prob.item()

    return real_prob, fake_prob