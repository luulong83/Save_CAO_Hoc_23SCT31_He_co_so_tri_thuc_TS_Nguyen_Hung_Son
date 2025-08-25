from model import analyze_image, DiseaseClassifier
import torch
import os

device = torch.device('cpu')
model = DiseaseClassifier().to(device)
model.load_state_dict(torch.load('trained_model.pth', map_location=device))
model.eval()

image_paths = [
    "chest_xray/test/NORMAL/IM-0001-0001.jpeg",
    "chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
]

for image_path in image_paths:
    if os.path.exists(image_path):
        result = analyze_image(image_path)
        print(f"Ảnh: {image_path}")
        print(f"Triệu chứng: {result['symptoms']}")
        if result["symptoms"]:
            print("Dự đoán: Có thể viêm phổi (PNEUMONIA)")
        else:
            print("Dự đoán: Bình thường (NORMAL)")
        # Debug thêm
        from PIL import Image
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
            pred = torch.softmax(output, dim=1)
            print(f"Xác suất PNEUMONIA: {pred[0][1]:.4f}")
    else:
        print(f"Không tìm thấy ảnh: {image_path}")