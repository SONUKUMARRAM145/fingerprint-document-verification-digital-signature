import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from model import FeatureExtractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FeatureExtractor().to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset_folder = "D:/datasets/SOCOFing/Altered/Altered-Easy"
save_folder = "backend_fingerprints"

os.makedirs(save_folder, exist_ok=True)

for filename in os.listdir(dataset_folder):
    if filename.endswith(".BMP") or filename.endswith(".bmp"):
        print(f"Processing {filename}...")
        path = os.path.join(dataset_folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load {filename}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(img)
        embedding = embedding.squeeze().cpu().numpy()
        np.save(os.path.join(save_folder, filename.split(".")[0]), embedding)

print("✅ Done!")