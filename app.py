import os
import streamlit as st
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    DistilBertTokenizerFast, DistilBertForSequenceClassification
)


LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

@st.cache_resource
def load_models():
    tfidf_model = joblib.load("model_tfidf_logreg.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    distil_model = DistilBertForSequenceClassification.from_pretrained("./distilbert_model")
    distil_tokenizer = DistilBertTokenizerFast.from_pretrained("./distilbert_model")
    bert_model = BertForSequenceClassification.from_pretrained("./bert_model")
    bert_tokenizer = BertTokenizerFast.from_pretrained("./bert_model")
    return tfidf_model, tfidf_vectorizer, distil_model, distil_tokenizer, bert_model, bert_tokenizer

@st.cache_data
def load_dataset():
    df = pd.read_csv("train.csv")
    df["nb_labels"] = df[LABELS].sum(axis=1)
    return df

tfidf_model, tfidf_vectorizer, distil_model, distil_tokenizer, bert_model, bert_tokenizer = load_models()
df = load_dataset()


st.title(" Détecteur de commentaires toxiques")

# Exploration du dataset
with st.expander("Explorer le dataset", expanded=False):
    st.markdown("### Exemple de lignes du dataset")
    st.dataframe(df.sample(5)[["comment_text"] + LABELS])

    st.markdown("### Distribution du nombre de labels par commentaire")
    nb_labels_count = df["nb_labels"].value_counts().sort_index()
    st.bar_chart(nb_labels_count)

    st.markdown("### Nombre de commentaires par label")
    label_counts = df[LABELS].sum().sort_values(ascending=False)
    st.bar_chart(label_counts)

    st.markdown("### Rechercher des exemples selon un label")
    selected_label = st.selectbox("Choisis une classe toxique :", LABELS)
    filtered = df[df[selected_label] == 1].sample(3)
    for i, row in filtered.iterrows():
        st.write(f" {row['comment_text'][:500]}...")

st.subheader(" Prédire la toxicité d’un commentaire")

model_choice = st.selectbox(
    "Choisis le modèle de prédiction :",
    ("TF-IDF + Logistic Regression", "DistilBERT", "BERT")
)

user_input = st.text_area(" Écris un commentaire (en anglais) :", "")

if st.button("Prédire la toxicité") and user_input.strip() != "":
    with st.spinner("Analyse en cours..."):

        if model_choice == "TF-IDF + Logistic Regression":
            vectorized = tfidf_vectorizer.transform([user_input])
            prediction = tfidf_model.predict_proba(vectorized)
            if isinstance(prediction, list):
                result = {LABELS[i]: round(pred[0], 3) for i, pred in enumerate(prediction)}
            else:
                prediction = np.array(prediction)
                result = {LABELS[i]: round(prediction[0][i], 3) for i in range(len(LABELS))}

        elif model_choice == "DistilBERT":
            inputs = distil_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
            with torch.no_grad():
                logits = distil_model(**inputs).logits
            probs = torch.sigmoid(logits).numpy()[0]
            result = {label: round(float(prob), 3) for label, prob in zip(LABELS, probs)}

        elif model_choice == "BERT":
            inputs = bert_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
            with torch.no_grad():
                logits = bert_model(**inputs).logits
            probs = torch.sigmoid(logits).numpy()[0]
            result = {label: round(float(prob), 3) for label, prob in zip(LABELS, probs)}

        st.success(" Résultat de la prédiction :")
        st.json(result)
else:
    st.info(" Entre un commentaire et clique sur le bouton pour lancer la prédiction.")

st.subheader("Comparaison des performances des modèles")

if os.path.exists("model_scores.csv"):
    score_df = pd.read_csv("model_scores.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["skyblue", "orange"]
    score_df.set_index("model")[["f1_micro", "f1_macro"]].plot.bar(ax=ax, color=colors)
    plt.title("Comparaison des F1-scores (micro vs macro)")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    st.pyplot(fig)

    best_model = score_df.sort_values("f1_macro", ascending=False).iloc[0]
    st.markdown(f" **Modèle le plus performant : `{best_model['model']}`** avec un F1-macro de **{best_model['f1_macro']:.3f}**.")
else:
    st.warning(" Fichier `model_scores.csv` non trouvé. Tu peux l'ajouter pour comparer les performances.")