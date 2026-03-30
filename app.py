import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# -------------------------------
# TITLE
# -------------------------------
st.title("📊 Customer Churn Prediction Dashboard")

# -------------------------------
# LOAD DATA (CACHED)
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Churn_Modelling.csv")
    return df

df = load_data()

# -------------------------------
# PREPROCESSING
# -------------------------------
def preprocess(df):
    X = df.drop(['Exited'], axis=1)
    y = df['Exited']

    X = X.drop(columns=['RowNumber','CustomerId','Surname'])
    X = pd.get_dummies(X, columns=['Geography','Gender'], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess(df)

# -------------------------------
# MODEL FUNCTIONS
# -------------------------------
@st.cache_resource
def train_logistic(X_train, y_train):
    model = LogisticRegression(class_weight={0:1,1:1.6})
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def train_tree(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def train_ann(X_train, y_train):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, verbose=0)

    return model

# -------------------------------
# EVALUATION FUNCTION
# -------------------------------
def evaluate(model, X_test, y_test, is_ann=False):
    if is_ann:
        y_prob = model.predict(X_test).ravel()
    else:
        y_prob = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    return fpr, tpr, precision, recall, auc, ap

# -------------------------------
# MODEL SELECTION
# -------------------------------
model_choice = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "Decision Tree", "ANN"]
)

# -------------------------------
# RUN MODEL
# -------------------------------
if model_choice == "Logistic Regression":
    model = train_logistic(X_train, y_train)
    fpr, tpr, precision, recall, auc, ap = evaluate(model, X_test, y_test)

elif model_choice == "Decision Tree":
    model = train_tree(X_train, y_train)
    fpr, tpr, precision, recall, auc, ap = evaluate(model, X_test, y_test)

elif model_choice == "ANN":
    model = train_ann(X_train, y_train)
    fpr, tpr, precision, recall, auc, ap = evaluate(model, X_test, y_test, is_ann=True)

# -------------------------------
# METRICS
# -------------------------------
st.subheader("📈 Model Performance")
st.write(f"**AUC Score:** {auc:.3f}")
st.write(f"**Average Precision:** {ap:.3f}")

# -------------------------------
# ROC CURVE
# -------------------------------
st.subheader("ROC Curve")

fig1, ax1 = plt.subplots()
ax1.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
ax1.plot([0,1], [0,1], linestyle='--')
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.legend()

st.pyplot(fig1)

# -------------------------------
# PRECISION-RECALL CURVE
# -------------------------------
st.subheader("Precision-Recall Curve")

fig2, ax2 = plt.subplots()
ax2.plot(recall, precision, label=f"AP = {ap:.2f}")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.legend()

st.pyplot(fig2)
