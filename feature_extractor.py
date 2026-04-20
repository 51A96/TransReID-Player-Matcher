import torch
from torchvision import transforms
from torchvision.models import resnet50

class FeatureExtractor:
    def __init__(self, model_config):
        self.model = self.load_model(model_config)
        self.model.eval()  # Set the model to evaluation mode

    def load_model(self, model_config):
        # Here you'd load the model according to the config provided.
        # This is a placeholder for model loading logic. Replace with actual.
        model = resnet50(pretrained=True)  # Example using ResNet50
        return model

    def preprocess(self, image):
        # Resize and normalize the image
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return preprocess(image)

    def extract_features(self, image):
        image = self.preprocess(image)
        image = image.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = self.model(image)  # Forward pass
        return features.flatten().numpy()  # Return normalized 768-dimensional vector