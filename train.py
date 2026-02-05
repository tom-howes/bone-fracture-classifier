import torch
import torch.nn as nn
from data import FractureDataset
from model import FractureCNN
from torchmetrics import F1Score, Precision, Recall
from torchvision import transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 32
# Transforms for train and validation datasets
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

train_dataset = FractureDataset('train/images', 'train/labels', train_transform)
val_dataset = FractureDataset('valid/images', 'valid/labels', val_transform)
test_dataset = FractureDataset('test/images', 'test/labels')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def train(epochs, lr=1e-3, weight_decay=1e-4):
        
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model = FractureCNN()
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Validation metrics
    f1 = F1Score(task='multilabel', num_labels=8).to(device)
    precision = Precision(task='multilabel', num_labels=8).to(device)
    recall = Recall(task='multilabel', num_labels=8).to(device)

    for epoch in range(epochs):
        ### Training phase

        model.train() # Set to train mode
        running_loss = 0
        for inputs, labels in train_loader:
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0) # Weight by batch size
            num_batches += 1

            if num_batches % 1 == 0:
                running_loss_avg = running_loss / (num_batches * BATCH_SIZE)
                print(f"Running Loss: {running_loss_avg:.4f}")
        train_loss = running_loss / len(train_loader) # final train avg
        
        ### Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():

            for inputs, labels in val_loader:
                outputs = model(inputs)
                predictions = nn.Sigmoid(outputs) > 0.5
                loss = criterion(predictions, labels)
                val_loss += loss.item()
                f1.update(predictions, labels)
                precision.update(predictions, labels)
                recall.update(predictions, labels)
            val_loss /= len(val_loader)
            f1.compute()
            precision.compute()
            recall.compute()

            # Print epoch summary
            print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")




                