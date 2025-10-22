import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

class CatDogClassifier:
    def __init__(self):
        # download  MobileNetV2
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.eval()
        
        # Transforming start image
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        # classes for cat and dog
        self.cat_classes = list(range(281, 286))  # cats in ImageNet
        self.dog_classes = list(range(151, 269))  # dogs in ImageNet

    def predict(self, image_path):
        """Predict who is on the image{cat, dog}"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_class = torch.max(probabilities, 1)
            
            predicted_class = top_class.item()
            confidence = top_prob.item()
            
            if predicted_class in self.cat_classes:
                return "Cat", confidence
            elif predicted_class in self.dog_classes:
                return "Dog", confidence
            else:
                return "Unknown", confidence
