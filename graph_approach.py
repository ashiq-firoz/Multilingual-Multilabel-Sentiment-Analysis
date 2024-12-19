import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

class SentimentGraphDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Create node features (embedding dimension = 768)
        x = input_ids.unsqueeze(-1).float()  # Shape: [seq_len, 1]
        
        # Create edges (sliding window approach)
        seq_length = len(input_ids)
        window_size = 2
        edges = []
        
        for i in range(seq_length):
            if attention_mask[i] == 0:  # Skip padding tokens
                continue
            for j in range(max(0, i-window_size), min(seq_length, i+window_size+1)):
                if i != j and attention_mask[j] == 1:
                    edges.append([i, j])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, y=torch.tensor(label))

class GNNSentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GNNSentimentClassifier, self).__init__()
        
        # Input embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Output layers
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial embedding
        x = self.embedding(x)
        
        # Apply GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # MLP classifier
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return x

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        
        # Training
        model.train()
        total_loss = 0
        predictions = []
        labels = []
        
        for batch in tqdm(train_loader, desc='Training'):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            
            predictions.extend(pred.cpu().numpy())
            labels.extend(batch.y.cpu().numpy())
        
        train_f1 = f1_score(labels, predictions, average='weighted')
        print(f'Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}, F1 = {train_f1:.4f}')
        
        # Validation
        val_f1 = evaluate(model, val_loader, device)
        print(f'Validation F1: {val_f1:.4f}')
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_f1_{val_f1:.3f}.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print('Early stopping!')
            break

def evaluate(model, loader, device):
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            
            predictions.extend(pred.cpu().numpy())
            labels.extend(batch.y.cpu().numpy())
    
    return f1_score(labels, predictions, average='weighted')

def main():
    # Load data
    print("Loading datasets...")
    train_df = pd.concat([pd.read_csv('Tam-SA-train.csv'),
                         pd.read_csv('cleaned_tamil_train.csv', header=0)], 
                        ignore_index=True)
    val_df = pd.concat([pd.read_csv('Tam-SA-val.csv'),
                       pd.read_csv('cleaned_tamil_dev.csv', header=0)], 
                      ignore_index=True)
    
    # Define label mapping
    label_map = {
        "Positive": 0,
        "Neutral": 1,
        "Negative": 2,
        "Mixed_feelings": 3,
        "unknown_state": 4,
        "Not Tulu": 5,
        "Mixed": 6,
        "not-Tamil":7,
    }

    # Convert text labels to numeric and handle any potential errors
    print("Converting labels to numeric values...")
    try:
        train_df['numeric_label'] = train_df['Label'].map(label_map)
        val_df['numeric_label'] = val_df['Label'].map(label_map)

        # Check for any NaN values after conversion
        if train_df['numeric_label'].isna().any() or val_df['numeric_label'].isna().any():
            print("Warning: Some labels couldn't be converted. Unique labels in data:")
            print("Train labels:", train_df['Label'].unique())
            print("Val labels:", val_df['Label'].unique())
            raise ValueError("Invalid labels found in dataset")

    except Exception as e:
        print(f"Error in label conversion: {e}")
        raise

    print(f"Number of training examples: {len(train_df)}")
    print(f"Number of validation examples: {len(val_df)}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
    
    # Create datasets
    train_dataset = SentimentGraphDataset(
        train_df['Text'].values,
        train_df['numeric_label'].values,
        tokenizer
    )
    val_dataset = SentimentGraphDataset(
        val_df['Text'].values,
        val_df['numeric_label'].values,
        tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNSentimentClassifier(
        input_dim=1,  # Changed from tokenizer.vocab_size
        hidden_dim=256,
        num_classes=len(label_map)
    ).to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main()
