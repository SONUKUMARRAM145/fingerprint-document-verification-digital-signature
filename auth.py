import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os
import cv2

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeatureExtractor().to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def authenticate_fingerprint(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = model(img).squeeze().cpu().numpy()

    best_score = -1
    best_user = None

    for filename in os.listdir("backend_fingerprints"):
        db_embedding = np.load(os.path.join("backend_fingerprints", filename))
        score = np.dot(query_embedding, db_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding))
        if score > best_score:
            best_score = score
            best_user = filename.split(".")[0]

    return best_user, best_score
