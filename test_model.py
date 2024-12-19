import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

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


def load_best_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

text = 'இனம் இனத்தோடு தான் சேரனும் வாழ்த்துக்கள் திரௌபதி'
label_map = {
    "Positive": 0,
    "Neutral": 1,
    "Negative": 2,
    "Mixed_feelings": 3,
    "unknown_state": 4,
    "Not Tulu": 5,
    "Mixed": 6
}
model, tokenizer = load_best_model('./best_model_f1_0.634')  # Load your pre-trained model
sentiment = predict_sentiment(text, model, tokenizer, label_map)  # Get prediction
print(f"Predicted sentiment: {sentiment}")