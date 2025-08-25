import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

# Định nghĩa model (ResNet18 pre-trained, fine-tune)
class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes=2):  # 0: NORMAL, 1: PNEUMONIA
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Hàm train_model có thêm tham số learning_rate
def train_model(data_dir='chest_xray', epochs=3, batch_size=8, learning_rate=0.0005, progress_bar=False):
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(f'{data_dir}/train', transform=transform)
    val_dataset = datasets.ImageFolder(f'{data_dir}/val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    device = torch.device('cpu')
    model = DiseaseClassifier().to(device)

    # Thêm class weights cho dataset không cân bằng
    class_weights = torch.tensor([1.0, 3.875/1.341]).to(device)  # NORMAL: 1.0, PNEUMONIA: ~2.89
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Dùng learning_rate truyền vào
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") if progress_bar else train_loader

        for inputs, labels in batch_iterator:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if progress_bar:
                batch_iterator.set_postfix(loss=running_loss / (batch_iterator.n + 1))

        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')

    # Save model
    torch.save(model.state_dict(), 'trained_model.pth')
    print('Model trained and saved as trained_model.pth')

# Hàm analyze_image (sử dụng model đã train)
def analyze_image(image_path):
    # Load model
    device = torch.device('cpu')
    model = DiseaseClassifier().to(device)
    model.load_state_dict(torch.load('trained_model.pth', map_location=device))
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load image
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)

    # Map to symptoms (giả lập dựa trên prediction)
    if pred[0][1] > 0.5:  # PNEUMONIA
        symptoms = ["cough", "fever", "chest_pain"]
    else:
        symptoms = []

    return {"symptoms": symptoms}