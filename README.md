# Naive Bayes Text Classification
## Overview
This project focuses on text classification using a Gaussian Naive Bayes model. The dataset consists of spam and ham messages, and the goal is to classify messages as either spam or not spam based on their content.

## Setup
### Prerequisites
Ensure you have the following installed:
* Python
* NLTK
* Scikit-learn
* Pandas
* Numpy
* Matplotlib

### Download NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## How to Run
### 1. Clone the Repository
```bash
git clone https://github.com/toan-ly/Naive-Bayes-Text-Classification.git
cd Naive-Bayes-Text-Classification
```

### 2. Preprocess the Data
Preprocess the text data using the functions defined in `preprocessing.py`

### 3. Train, Evaluate the Model, and Predict New Messages
Run the following script to train and evaluate the Gaussian Naive Bayes model:
```bash
python main.py
```

## Example Prediction
```python
test_input = 'Urgent! Please sign in to verify your account!'
pred_cls = predict.predict(test_input, model, dictionary, le)
print(f'Prediction: {pred_cls}')
```

