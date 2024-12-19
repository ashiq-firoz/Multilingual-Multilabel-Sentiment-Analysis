# 🌍 Multilingual-Multilabel Sentiment Analysis  
A deep learning project for analyzing multilingual text sentiments (Tamil and Tulu) using transformers and graph-based approaches.  

---

## ✨ Features  
- 🔄 **Fine-tuning XLM-RoBERTa** for multilingual sentiment classification.  
- 🌐 **Graph Neural Networks (GNN)** for sentiment analysis.  
- 🎭 Support for multiple sentiment classes: Positive, Negative, Neutral, Mixed Feelings, etc.  
- ⏱️ **Training with Early Stopping** and Model Checkpointing.  
- 📊 **F1-Score Based Evaluation** for robust performance.  

---

## ⚙️ Installation  

### 🐍 Using Conda  
1. **Create a new conda environment**:  
   ```bash  
   conda create -n sentiment python=3.9  
   conda activate sentiment  
   ```  
2. **Install basic dependencies**:  
   ```bash  
   conda install pytorch torchvision -c pytorch  
   pip install -r requirements.txt  
   ```  
3. **For graph-based approaches**, install additional dependencies:  
   ```bash  
   pip install -r graph_requirements.txt  
   ```  

### 📦 Using pip only  
Install all required packages:  
```bash  
pip install -r requirements.txt  
pip install -r graph_requirements.txt  # Optional: for graph-based approach  
```  

---

## 📂 Dataset Structure  
Ensure your dataset files are in the `dataset/` folder:  
- `cleaned_tamil_dev.csv`  
- `cleaned_tamil_train.csv`  
- `Tam-SA-train.csv`  
- `Tam-SA-val.csv`  
- `Tulu_SA_train.csv`  
- `Tulu_SA_val.csv`  

---

## 🚀 Usage  

### Training the Models  
#### For transformer-based approach:  
```bash  
python main.py  
```  

#### For graph-based approach:  
```bash  
python graph_approach.py  
```  

### Testing a Model  
```bash  
python test_model.py  
```  

💡 *Use this command to test the trained model on new text.*  

---

## 🛠️ Model Architectures  
- **Transformer Approach**: Fine-tuned XLM-RoBERTa base model.  
- **Graph Approach**: Graph Convolutional Networks (GCN) with custom tokenization.  

---

## 📈 Performance  
- 🤖 **Transformer Model**: ~0.63 F1 score.  
- 🌐 **Graph Model**: Comparable performance with unique strengths.  

---

## 📁 Project Structure  
```plaintext  
.  
├── dataset/                  # Data files  
├── main.py                  # Transformer-based implementation  
├── graph_approach.py        # GNN-based implementation  
├── test_model.py            # Model inference code
├── FInetuning_ XLM-RoBERTa.ipynb         #Jupyter notebook
├── requirements.txt         # Basic dependencies  
└── graph_requirements.txt   # Additional GNN dependencies 

```  

---

## 💻 Hardware Used  
- NVIDIA 4060 GPU with **8GB RAM**.  
- CUDA 12.1 and PyTorch 2.5.1.  
- 🖥️ Tested on **Google Colab T4 GPU** (15GB GPU RAM).  



