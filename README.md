# 🧠 Détection de commentaires toxiques (TP NLP)

Ce projet est réalisé dans le cadre d’un TP en groupe sur le **traitement automatique du langage (NLP)** avec une interface Streamlit. L’objectif est de **classifier des commentaires en ligne toxiques** à partir du dataset **[Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)** de Kaggle.

---

## 📦 Dataset

- 📁 Source : [Kaggle - Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- 📝 Tâche : classification **multilabel** (un commentaire peut appartenir à plusieurs catégories)
- 📊 Labels :  
  - `toxic`  
  - `severe_toxic`  
  - `obscene`  
  - `threat`  
  - `insult`  
  - `identity_hate`

---

## 🧪 Modèles utilisés

### 1. `TF-IDF + LogisticRegression`
- Modèle de base rapide et léger
- Entraîné avec `OneVsRestClassifier`
- ✨ Avantage : temps de prédiction très court

### 2. `DistilBERT`
- Modèle de Transfert Learning léger
- Pré-entraîné par HuggingFace
- Fine-tuning complet avec PyTorch + HuggingFace `Trainer`
- ⚖️ Excellent compromis entre vitesse et performance

### 3. `BERT base`
- Modèle pré-entraîné `bert-base-uncased`
- Fine-tuning complet
- 🧠 Très performant, mais plus lourd à exécuter

---

## 🚀 Application Streamlit

### Fonctionnalités :
- Interface interactive pour saisir un commentaire
- Prédiction en temps réel avec le modèle sélectionné
- Visualisation des probabilités par classe
- Comparaison graphique des performances (F1-micro / F1-macro)

### Lancement local :

```bash
streamlit run app.py
![alt text](image.png)