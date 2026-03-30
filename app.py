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
model_options = ['Logistic Regression', 'Decision Tree', 'Keras ANN']
selected_models = st.sidebar.multiselect("Select models to compare:", model_options, default=model_options)

# --- 6. Main Content Area (Tabs) ---
if not selected_models:
    st.warning("👈 Please select at least one model from the sidebar to view metrics.")
else:
    tab1, tab2, tab3 = st.tabs(["📊 Model Metrics Comparison", "📈 ROC Curve", "📉 Precision-Recall Curve"])
    
    # Calculations for selected models
    metrics_data = []
    roc_data = {}
    pr_data = {}
    
    # 🛠️ THE FIX: Flatten y_test to ensure Scikit-Learn accepts it safely from the Streamlit cache
    y_test_flat = np.array(y_test).flatten()
    
    for model_name in selected_models:
        y_prob = predictions[model_name]['proba']
        y_pred_class = predictions[model_name]['class']
        
        # Metrics using the flattened y_test
        acc = accuracy_score(y_test_flat, y_pred_class)
        auc = roc_auc_score(y_test_flat, y_prob)
        ap = average_precision_score(y_test_flat, y_prob)
        metrics_data.append({'Model': model_name, 'Accuracy': acc, 'AUC': auc, 'Average Precision': ap})
        
        # ROC Data
        fpr, tpr, _ = roc_curve(y_test_flat, y_prob)
        roc_data[model_name] = (fpr, tpr, auc)
        
        # PR Data
        prec, rec, _ = precision_recall_curve(y_test_flat, y_prob)
        pr_data[model_name] = (prec, rec, ap)

    # TAB 1: Metrics Table
    with tab1:
        st.subheader("Comparison Table")
        comparison_df = pd.DataFrame(metrics_data).set_index('Model')
        st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))
        
        st.markdown("""
        **Quick Guide:**
        * **Accuracy:** Overall correctness (Can be misleading in imbalanced datasets like churn).
        * **AUC:** How well the model separates churners from non-churners. Higher is better.
        * **Average Precision:** Area under the PR curve. Crucial when focusing on finding the actual churners accurately.
        """)

    # TAB 2: ROC Curve
    with tab2:
        st.subheader("Receiver Operating Characteristic (ROC) Curve")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name, (fpr, tpr, auc) in roc_data.items():
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
            
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.set_title('ROC Curve for Churn Prediction Models')
        ax.legend(loc='lower right')
        ax.grid(True)
        
        st.pyplot(fig)

    # TAB 3: Precision-Recall Curve
    with tab3:
        st.subheader("Precision-Recall Curve")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        for model_name, (prec, rec, ap) in pr_data.items():
            ax2.plot(rec, prec, label=f'{model_name} (AP = {ap:.2f})')
            
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve for Churn Prediction Models')
        ax2.legend(loc='lower left')
        ax2.grid(True)
        
        st.pyplot(fig2)
