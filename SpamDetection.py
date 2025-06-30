import numpy as np,pandas as pd,matplotlib.pyplot as plt
import math
from collections import Counter

df=pd.read_csv("LogisticRegression\spam.csv", encoding='latin-1')
df=df[['v1','v2']]
df.columns=['label','message']
# print(df.head())

#Text Clean Up
import re
def cleanText(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text
df['clean_message']=df['message'].apply(cleanText)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
# print(df.head())

#Text Vectorization
all_words=[]
for msg in df['clean_message']:
    all_words += msg.split()

vocab_size = 1000
vocab = Counter(all_words).most_common(vocab_size)
vocab = [word for word, freq in vocab]

# print(vocab)
word_to_index = {word: i for i, word in enumerate(vocab)}

#Message to Vector
def message_to_vector(msg):
    vector = np.zeros(vocab_size)
    for word in msg.split():
        if word in word_to_index:
            vector[word_to_index[word]] += 1
    return vector

X = np.array([message_to_vector(msg) for msg in df['clean_message']])
y = df['label_num'].values

# print("Shape of X:", X.shape)
# print("Shape of y:", y.shape)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def compute_cost(w,X,y,b):
    m=X.shape[0]
    cost=0
    for i in range(m):
        f_wb=sigmoid(np.dot(w,X[i])+b)
        cost += -y[i] * np.log(f_wb + 1e-15) - (1 - y[i]) * np.log(1 - f_wb + 1e-15)
    total_cost=(cost*-1)/m  
    return total_cost

def compute_gradient(w,X,y,b):
    m=X.shape[0]
    dj_dw=0
    dj_db=0
    for i in range(m):
        f_wb=sigmoid(np.dot(w,X[i])+b)
        dj_dw+=(f_wb-y[i])*X[i]
        dj_db+=(f_wb-y[i])
        
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

def gradient_descent(w,X,y,b,alpha,iterations,compute_cost,compute_gradient):
    J_history = []
    w_history = []
    
    for i in range(iterations):
        
        dj_dw,dj_db=compute_gradient(w,X,y,b)
        
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
        if i < 100000:      # prevent resource exhaustion 
            cost =  compute_cost(w,X,y,b)
            J_history.append(cost)
        if i % math.ceil(iterations/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")
            
    return w, b, J_history, w_history #return w and J,w history for graphing

w = np.zeros(X.shape[1])  # 1000 features
b = 0
alpha = 0.1
iterations = 1000

w, b, cost_history, _ = gradient_descent(w, X, y, b, alpha, iterations, compute_cost, compute_gradient)

def predict(X, w, b, threshold=0.5):
    preds = []
    for i in range(X.shape[0]):
        z = np.dot(w, X[i]) + b
        prob = sigmoid(z)
        preds.append(1 if prob >= threshold else 0)
    return np.array(preds)

y_pred = predict(X, w, b)
accuracy = np.mean(y_pred == y) * 100
print(f"\nTraining Accuracy: {accuracy:.2f}%")

def classify_message(msg):
    msg = cleanText(msg)
    vector = message_to_vector(msg)
    prob = sigmoid(np.dot(w, vector) + b)
    return "Spam" if prob >= 0.5 else "Ham"

print(classify_message("Congratulations! You've won a free ticket. Call now!"))
print(classify_message("Hey, will call you later."))
