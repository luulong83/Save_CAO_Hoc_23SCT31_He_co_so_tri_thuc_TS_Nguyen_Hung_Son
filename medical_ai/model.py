import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from collections import Counter
from efficientnet_pytorch import EfficientNet

class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

def train_model(data_dir='chest_xray', epochs=10, batch_size=5, learning_rate=0.0005, progress_bar=False):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(f'{data_dir}/train', transform=transform_train)
    val_dataset = datasets.ImageFolder(f'{data_dir}/val', transform=transform_val)  # Sửa từ val_val thành transform_val
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiseaseClassifier().to(device)

    # Tính trọng số lớp động dựa trên dataset
    class_counts = Counter(train_dataset.targets)
    num_normal = class_counts[0]  # NORMAL
    num_pneumonia = class_counts[1]  # PNEUMONIA
    class_weights = torch.tensor([num_pneumonia / num_normal, num_normal / num_pneumonia]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }

    best_val_loss = float('inf')
    patience = 3
    counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") if progress_bar else train_loader

        for inputs, labels in batch_iterator:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if progress_bar:
                batch_iterator.set_postfix(loss=running_loss / (batch_iterator.n + 1))

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        print(f'Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    torch.save(model.state_dict(), 'trained_model.pth')
    print('Model trained and saved as trained_model.pth')
    return history

def analyze_image(image_path):
    device = torch.device('cpu')
    model = DiseaseClassifier().to(device)
    model.load_state_dict(torch.load('trained_model.pth', map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)

    prob_pneumonia = pred[0][1].item()
    prob_normal = pred[0][0].item()
    threshold = 0.5
    if prob_pneumonia >= threshold:
        diagnosis = "PNEUMONIA"
        symptoms = ["cough", "fever", "chest_pain"] if prob_pneumonia > 0.9 else []  # Chỉ gán triệu chứng nếu rất chắc chắn
    else:
        diagnosis = "NORMAL"
        symptoms = []

    return {
        "diagnosis": diagnosis,
        "prob_normal": prob_normal,
        "prob_pneumonia": prob_pneumonia,
        "symptoms": symptoms
    }