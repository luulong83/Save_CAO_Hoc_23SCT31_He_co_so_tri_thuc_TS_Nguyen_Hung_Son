from model import analyze_image, DiseaseClassifier
import torch
import os

device = torch.device('cpu')
model = DiseaseClassifier().to(device)
model.load_state_dict(torch.load('trained_model.pth', map_location=device))
model.eval()

image_paths = [
    "chest_xray/test/NORMAL/NORMAL2-IM-0374-0001-0002.jpeg",
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
        print(f"Xác suất PNEUMONIA: {result['prob_pneumonia']:.4f}")
    else:
        print(f"Không tìm thấy ảnh: {image_path}")