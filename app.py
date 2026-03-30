import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, accuracy_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

# --- 1. Page Configuration ---
st.set_page_config(page_title="Churn Model Comparator", layout="wide")
st.title("🏦 Bank Churn Prediction: Model Comparison")
st.markdown("""
Compare how traditional Machine Learning models stack up against a Deep Learning ANN when predicting customer churn. 
Use the sidebar to select models and view their performance curves!
""")

# --- 2. Data Loading & Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv('Churn_Modelling.csv')
    
    # Drop irrelevant columns
    X = df.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname'], axis=1)
    Y = df['Exited']
    
    # One-Hot Encoding
    X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Scaling
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

X_train, X_test, y_train, y_test = load_and_preprocess_data()

# --- 3. Model Training ---
@st.cache_resource
def train_models(X_tr, y_tr):
    models = {}
    
    # 1. Logistic Regression
    lr = LogisticRegression(class_weight={0:1, 1:1.6})
    lr.fit(X_tr, y_tr)
    models['Logistic Regression'] = lr
    
    # 2. Decision Tree
    dt = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5, min_samples_split=2)
    dt.fit(X_tr, y_tr)
    models['Decision Tree'] = dt
    
    # 3. Keras ANN
    ann = Sequential()
    ann.add(Input(shape=(X_tr.shape[1],)))
    ann.add(Dense(3, activation='relu'))
    ann.add(Dense(4, activation='relu'))
    ann.add(Dense(1, activation='sigmoid'))
    ann.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Verbose=0 keeps the Streamlit terminal clean
    ann.fit(X_tr, y_tr, epochs=100, validation_split=0.2, verbose=0) 
    models['Keras ANN'] = ann
    
    return models

with st.spinner('Training models... (This might take a few seconds on the first run)'):
    trained_models = train_models(X_train, y_train)

# --- 4. Generating Predictions ---
@st.cache_data
def get_predictions(_models, X_te):
    preds = {}
    
    # Logistic Regression Predictions
    preds['Logistic Regression'] = {
        'proba': _models['Logistic Regression'].predict_proba(X_te)[:, 1],
        'class': _models['Logistic Regression'].predict(X_te)
    }
    
    # Decision Tree Predictions
    preds['Decision Tree'] = {
        'proba': _models['Decision Tree'].predict_proba(X_te)[:, 1],
        'class': _models['Decision Tree'].predict(X_te)
    }
    
    # ANN Predictions
    ann_proba = _models['Keras ANN'].predict(X_te).flatten()
    preds['Keras ANN'] = {
        'proba': ann_proba,
        'class': np.where(ann_proba > 0.5, 1, 0)
    }
    
    return preds

predictions = get_predictions(trained_models, X_test)

# --- 5. Sidebar UI ---
st.sidebar.header("Model Selection")
model_options = ['Logistic Regression', 'Decision Tree', 'Keras ANN
