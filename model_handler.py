import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image


class SignLanguageModel:
    def __init__(self, model_path="./model"):
        print(f"Loading model from {model_path}...")
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        print("Model loaded successfully!")

    def analyze_sign(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        inputs = self.processor(images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()

        label = self.model.config.id2label[predicted_class]
        confidence = torch.softmax(logits, dim=-1)[0][predicted_class].item()

        return f"This sign means: {label} (confidence: {confidence:.2%})"