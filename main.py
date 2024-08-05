import string
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import preprocessing as pp
import predict

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Read dataset
DATASET_PATH = 'data/2cls_spam_text_cls.csv'
df = pd.read_csv(DATASET_PATH)
print(df)

# Extract messages and labels
messages = df['Message'].values.tolist()
labels = df['Category'].values.tolist()

# Preprocess messages
messages = pp.preprocess_messages(messages)
dictionary = pp.create_dictionary(messages)
X = np.array([pp.create_features(tokens, dictionary) for tokens in messages])

# One hot encode labels
le = LabelEncoder()
y = le.fit_transform(labels)
print(f'Classes: {le.classes_}')
print(f'Encoded labels: {y}')

# Split data
VAL_SIZE = 0.2
TEST_SIZE = 0.125
SEED = 0

X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                  test_size=VAL_SIZE, 
                                                  shuffle=True, 
                                                  random_state=SEED)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size=TEST_SIZE,
                                                    shuffle=True,
                                                    random_state=SEED)

# Train model
model = GaussianNB()
print('Start training...')
model = model.fit(X_train, y_train)
print('Training completed!')

# Evaluate
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f'Val accuracy: {val_acc}')
print(f'Test accuracy: {test_acc}')

# Predict
test_input = 'I am actually thinking a way of doing something useful'
pred_cls = predict.predict(test_input, model, dictionary, le)
print(f'Prediction: {pred_cls}')

test_input = 'Urgent! Please sign in to verify your account!'
pred_cls = predict.predict(test_input, model, dictionary, le)
print(f'Prediction: {pred_cls}')




