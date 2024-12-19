import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
#import wandb  # for experiment tracking

# Load the datasets
print("Loading datasets...")
train_df = pd.concat([
    pd.read_csv('dataset/Tam-SA-train.csv'),
    pd.read_csv('dataset/cleaned_tamil_dev.csv', header=0),
    pd.read_csv('dataset/Tulu_SA_train.csv', header=0)
], ignore_index=True)

val_df = pd.concat([
    pd.read_csv('dataset/Tam-SA-val.csv'),
    pd.read_csv('dataset/cleaned_tamil_dev.csv', header=0),
    pd.read_csv('dataset/Tulu_SA_val.csv', header=0)
], ignore_index=True)

# train_df = pd.read_csv("Tulu_SA_train.csv")
# val_df = pd.read_csv("Tulu_SA_val.csv")

# Define label mapping   (Change this based on your dataset and target)
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

# Custom Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])  # Convert to int

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize model and tokenizer
print("Initializing model and tokenizer...")
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_map),
    problem_type="single_label_classification",
    ignore_mismatched_sizes=True
)

# Create datasets using the numeric labels
train_dataset = SentimentDataset(
    train_df['Text'].values,
    train_df['numeric_label'].values,  # Use numeric labels
    tokenizer
)
val_dataset = SentimentDataset(
    val_df['Text'].values,
    val_df['numeric_label'].values,  # Use numeric labels
    tokenizer
)

# Set device and batch size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

batch_size = 16 if torch.cuda.is_available() else 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Move model to device
model = model.to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
num_epochs = 10
num_training_steps = num_epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_training_steps // 10,
    num_training_steps=num_training_steps
)

# Training function
def train_model():
    best_val_f1 = 0
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')

        # Training phase
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_labels = []

        for batch in tqdm(train_loader, desc='Training'):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            predictions = torch.argmax(outputs.logits, dim=-1)
            train_predictions.extend(predictions.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        train_f1 = f1_score(train_labels, train_predictions, average='weighted')

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_val_loss += loss.item()

                predictions = torch.argmax(outputs.logits, dim=-1)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(val_labels, val_predictions, average='weighted')

        print(f'Training Loss: {avg_train_loss:.3f}, Training F1: {train_f1:.3f}')
        print(f'Validation Loss: {avg_val_loss:.3f}, Validation F1: {val_f1:.3f}')

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            model_save_path = f'best_model_f1_{val_f1:.3f}'
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            print(f'Saved best model with F1: {val_f1:.3f}')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

def predict_sentiment(text, model, tokenizer, label_map):
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to the appropriate device

    # Tokenize the input text and move tensors to the same device as the model
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)  # Get model outputs
        prediction = torch.argmax(outputs.logits, dim=-1)  # Find the predicted class

    # Convert numeric prediction back to text label
    reverse_label_map = {v: k for k, v in label_map.items()}  # Reverse the label mapping
    return reverse_label_map[prediction.item()]

# Train the model
print("Starting training...")
train_model()

# # Load best model for inference
# def load_best_model(model_path):
#     model = AutoModelForSequenceClassification.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     return model, tokenizer

# text = 'இனம் இனத்தோடு தான் சேரனும் வாழ்த்துக்கள் திரௌபதி'
# label_map = {
#     "Positive": 0,
#     "Neutral": 1,
#     "Negative": 2,
#     "Mixed_feelings": 3,
#     "unknown_state": 4,
#     "Not Tulu": 5,
#     "Mixed": 6
# }
# model, tokenizer = load_best_model('./best_model_f1_0.574')  # Load your pre-trained model
# sentiment = predict_sentiment(text, model, tokenizer, label_map)  # Get prediction
# print(f"Predicted sentiment: {sentiment}")