import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import os

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if torch.cuda.is_available():
    print(f" Using GPU: {torch.cuda.get_device_name(0)}")
   
    print("  Using CPU")
print()

# Configuration
WINDOW_SIZE = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 200
PATIENCE = 15
STRIDE = 1




def load_and_analyze_dataset(filepath, dataset_name):
    
    try:
        data = pd.read_csv(filepath).astype(np.float32)
        
        labels = data.iloc[:, -1].astype(np.int64)
        features = data.iloc[:, :-1]
        
       
        label_counts = Counter(labels)
        print(f"Class distribution:")
        for class_id in sorted(label_counts.keys()):
            print(f"   Class {class_id}: {label_counts[class_id]} samples ({label_counts[class_id]/len(labels)*100:.1f}%)")
        
        return features, labels, label_counts
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None, None, None


train_features, train_labels, train_class_dist = load_and_analyze_dataset(
    'sonata_combined_attacks_preprocessed.csv', 'Sonata (Training Dataset)'
)

print()

test_features, test_labels, test_class_dist = load_and_analyze_dataset(
    'combined_attacks_preprocessed.csv', 'car hacking (Testing Dataset)'
)

if train_features is None or test_features is None:
    print(" Failed to load datasets")
    exit(1)

print()


if train_features.shape[1] != test_features.shape[1]:
    print("  Feature dimensions don't match!")
   
    min_features = min(train_features.shape[1], test_features.shape[1])
    train_features = train_features.iloc[:, :min_features]
    test_features = test_features.iloc[:, :min_features]
    
else:
    print("Feature dimensions match")


train_classes = set(train_labels)
test_classes = set(test_labels)
common_classes = train_classes.intersection(test_classes)
train_only_classes = train_classes - test_classes
test_only_classes = test_classes - train_classes

NUM_CLASSES = max(max(train_classes), max(test_classes)) + 1


def create_train_val_split(features, labels, val_ratio=0.2):
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, 
        test_size=val_ratio, 
        stratify=labels, 
        random_state=42
    )
    
    return X_train, X_val, y_train, y_val

set_seed(42)


features_train, features_val, labels_train, labels_val = create_train_val_split(
    train_features, train_labels
)


train_label_counts = Counter(labels_train)



val_label_counts = Counter(labels_val)




features_min = features_train.min()
features_max = features_train.max()


features_train_normalized = (features_train - features_min) / (features_max - features_min + 1e-8)
features_val_normalized = (features_val - features_min) / (features_max - features_min + 1e-8)
features_test_normalized = (test_features - features_min) / (features_max - features_min + 1e-8)



def create_sequences_with_stride(features, labels, window_size, stride):
    
    sequences = []
    sequence_labels = []
    
    
    
    for i in range(0, len(features) - window_size + 1, stride):
        sequence = features.iloc[i:i + window_size].values
        
        
        window_labels = labels.iloc[i:i + window_size] if hasattr(labels, 'iloc') else labels[i:i + window_size]
        
       
        window_label = 0  
        
        
        for label in window_labels:
            if label != 0:  
                window_label = label
                break  
        
        sequences.append(sequence)
        sequence_labels.append(window_label)
    
    sequences = np.array(sequences)
    sequence_labels = np.array(sequence_labels)
    
    
    
    seq_label_counts = Counter(sequence_labels)
    for class_id in sorted(seq_label_counts.keys()):
        count = seq_label_counts.get(class_id, 0)
        percentage = (count / len(sequence_labels)) * 100
        
    
    return sequences, sequence_labels

X_train_seq, y_train_seq = create_sequences_with_stride(features_train_normalized, labels_train, WINDOW_SIZE, STRIDE)

X_val_seq, y_val_seq = create_sequences_with_stride(features_val_normalized, labels_val, WINDOW_SIZE, STRIDE)

X_test_seq, y_test_seq = create_sequences_with_stride(features_test_normalized, test_labels, WINDOW_SIZE, STRIDE)


X_train = torch.tensor(X_train_seq, dtype=torch.float32)
y_train = torch.tensor(y_train_seq, dtype=torch.long)
X_val = torch.tensor(X_val_seq, dtype=torch.float32)
y_val = torch.tensor(y_val_seq, dtype=torch.long)
X_test = torch.tensor(X_test_seq, dtype=torch.float32)
y_test = torch.tensor(y_test_seq, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


train_seq_class_counts = Counter(y_train_seq)
total_train_seq = len(y_train_seq)


class_weights_sqrt = []
for class_id in range(NUM_CLASSES):
    if class_id in train_seq_class_counts:
        weight = np.sqrt(total_train_seq / train_seq_class_counts[class_id])
        class_weights_sqrt.append(weight)
    else:
        class_weights_sqrt.append(1.0)

class_weights = torch.tensor(class_weights_sqrt, dtype=torch.float32).to(device)


for class_id in range(NUM_CLASSES):
    count = train_seq_class_counts.get(class_id, 0)
    percentage = (count / total_train_seq) * 100 if total_train_seq > 0 else 0
    weight = class_weights_sqrt[class_id]
    


class FocalLoss(nn.Module):
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SequenceTransformerClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes, d_model=64, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        
        self.positional_encoding = nn.Parameter(torch.randn(seq_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)

        
        x = self.input_projection(x)

        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

       
        x = x + self.positional_encoding.unsqueeze(0)[:, :x.size(1), :]
        x = self.dropout(x)

       
        x = self.transformer_encoder(x)

        cls_output = x[:, 0, :]

        return self.classifier(cls_output)


input_dim = X_train.shape[2]
seq_len = X_train.shape[1]

model = SequenceTransformerClassifier(
    input_dim=input_dim,
    seq_len=seq_len,
    num_classes=NUM_CLASSES,
    d_model=64,
    nhead=2,
    num_layers=2,
    dropout=0.15
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

criterion = FocalLoss(alpha=class_weights, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)



train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_balanced_accuracies = []
val_balanced_accuracies = []
best_val_balanced_acc = 0.0
patience_counter = 0
start_time = time.time()

def calculate_balanced_accuracy(model, loader, device):
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return balanced_accuracy_score(all_targets, all_predictions)

for epoch in range(EPOCHS):
    
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train
    train_balanced_acc = calculate_balanced_accuracy(model, train_loader, device)
    
    
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = correct_val / total_val
    val_balanced_acc = calculate_balanced_accuracy(model, val_loader, device)
    
    
    scheduler.step()
    
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    train_balanced_accuracies.append(train_balanced_acc)
    val_balanced_accuracies.append(val_balanced_acc)
    
   
    if (epoch + 1) % 10 == 0 or epoch < 5:
        print(f"Epoch [{epoch+1}/{EPOCHS}] - "
              f"Train Loss: {train_loss:.4f}, Train Bal Acc: {train_balanced_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Bal Acc: {val_balanced_acc:.4f}")
    
    # Early stopping based on balanced accuracy
    if val_balanced_acc > best_val_balanced_acc:
        best_val_balanced_acc = val_balanced_acc
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_balanced_acc': val_balanced_acc,
            'class_weights': class_weights.cpu(),
            'normalization_params': {
                'min': features_min,
                'max': features_max
            }
        }, 'best_cross_dataset_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f" Early stopping triggered after {epoch+1} epochs")
            break

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")



def evaluate_cross_dataset(model, test_loader, device, num_classes):
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    
    test_acc = correct / total
    balanced_acc = balanced_accuracy_score(all_targets, all_predictions)
    
    return test_acc, balanced_acc, all_predictions, all_targets, np.array(all_probabilities)


checkpoint = torch.load('best_cross_dataset_model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(" Best model loaded for cross-dataset testing")


test_acc, balanced_test_acc, predictions, targets, probabilities = evaluate_cross_dataset(
    model, test_loader, device, NUM_CLASSES
)



target_classes = [1, 2, 3]
class_accuracies = {}

for class_id in target_classes:
    class_mask = np.array(targets) == class_id
    class_predictions = np.array(predictions)[class_mask]
    class_targets = np.array(targets)[class_mask]
    
    if len(class_targets) > 0:
        class_accuracy = (class_predictions == class_targets).sum() / len(class_targets)
        class_accuracies[class_id] = class_accuracy
    else:
        class_accuracies[class_id] = 0.0

for class_id in target_classes:
    accuracy = class_accuracies[class_id]
    print(f" Class {class_id}: {accuracy:.4f} ({accuracy*100:.2f}%)")

