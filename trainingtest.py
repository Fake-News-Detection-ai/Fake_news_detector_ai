# =========================================
# 🧠 FAKE NEWS DETECTION USING SVM (80/20 Split)
# with Accuracy, Precision, Recall, F1, AUROC
# + 4-Step Progress Bar + Save Model (.pkl)
# =========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

# 1️⃣ Load dataset
print("📂 Loading dataset...")
# Prefer the pre-vectorized file if it exists, otherwise fall back to raw dataset
csv_candidates = [
    "C:/Users/kenti/AI/Fake_news_detector_ai/PreparingDATA/vectorized_data.csv",
    "C:/Users/kenti/AI/Fake_news_detector_ai/Documents/fake_news_dataset.csv",
    "Documents/fake_news_dataset.csv",
]
data = None
for path in csv_candidates:
    try:
        data = pd.read_csv(path)
        print(f"Loaded: {path}")
        break
    except Exception:
        continue
if data is None:
    raise FileNotFoundError("No dataset found. Checked: " + ", ".join(csv_candidates))

# 2️⃣ Split features and labels (handle both vectorized and raw text datasets)
if 'label' not in data.columns:
    raise KeyError("Dataset must contain a 'label' column")

y = data['label']
X_df = data.drop(columns=['label'])

# If any column is non-numeric, combine text columns and vectorize
vectorizer = None
if X_df.select_dtypes(include=['object']).shape[1] > 0:
    text_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    print(f"🔄 Detected text columns: {text_cols} — vectorizing them with CountVectorizer")
    combined_text = X_df[text_cols].astype(str).agg(' '.join, axis=1)
    vectorizer = CountVectorizer(max_features=3000, stop_words='english')
    X = vectorizer.fit_transform(combined_text)
else:
    # All numeric → use as-is
    X = X_df.values

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"✅ Dataset ready → Training: {X_train.shape[0]} | Testing: {X_test.shape[0]}")

# 4️⃣ Initialize SVM model
svm_model = SVC(kernel='linear', probability=True, random_state=42)

# 5️⃣ Multi-step progress bar
steps = ["Training model", "Predicting", "Evaluating metrics", "Saving model"]
with tqdm(total=len(steps), desc="🧠 Overall Progress", ncols=80, colour='cyan') as pbar:

    # Step 1 — Train
    pbar.set_description("🚀 Training SVM model")
    svm_model.fit(X_train, y_train)
    pbar.update(1)

    # Step 2 — Predict
    pbar.set_description("🔍 Predicting on test data")
    y_pred = svm_model.predict(X_test)
    y_proba = svm_model.predict_proba(X_test)[:, 1]
    pbar.update(1)

    # Step 3 — Evaluate
    pbar.set_description("📊 Evaluating metrics")
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='REAL')
    recall    = recall_score(y_test, y_pred, pos_label='REAL')
    f1        = f1_score(y_test, y_pred, pos_label='REAL')
    auroc     = roc_auc_score((y_test == 'REAL').astype(int), y_proba)
    pbar.update(1)

    # Step 4 — Save model
    pbar.set_description("💾 Saving model to disk")
    model_path = "C:/Users/kenti/AI/Fake_news_detector_ai/svm_fake_news_model.pkl"
    joblib.dump(svm_model, model_path)
    pbar.update(1)

# 6️⃣ Display metrics
print("\n✅ All steps completed successfully!\n")
print("📊 Model Performance Metrics (SVM)")
print("----------------------------------")
print(f"✅ Accuracy : {accuracy:.3f}")
print(f"🎯 Precision: {precision:.3f}")
print(f"📈 Recall   : {recall:.3f}")
print(f"🏆 F1-Score : {f1:.3f}")
print(f"💠 AUROC    : {auroc:.3f}")
print(f"\n💾 Model saved successfully at:\n{model_path}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# 7️⃣ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['FAKE','REAL'], yticklabels=['FAKE','REAL'])
plt.title("Confusion Matrix – SVM Fake News Detection (80/20 Split)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 8️⃣ ROC Curve
RocCurveDisplay.from_predictions((y_test == 'REAL').astype(int), y_proba)
plt.title("ROC Curve – SVM Fake News Detection")
plt.show()
