# [file name]: baselineD_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class BaselineDModel:
    def __init__(self):
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.model_type = "Combined Embeddings + Similarities"
        self.models = {}
        self.scaler = StandardScaler()

    def load_split_data(self, train_path='baseline_train.csv', val_path='baseline_val.csv', test_path='baseline_test.csv'):
        """Load the train/val/test split datasets"""
        print("📁 Loading train/val/test splits...")
        try:
            self.train_df = pd.read_csv(train_path)
            self.val_df = pd.read_csv(val_path)
            self.test_df = pd.read_csv(test_path)
            
            print(f"✅ Data loaded successfully:")
            print(f"   Train: {len(self.train_df)} samples")
            print(f"   Val: {len(self.val_df)} samples") 
            print(f"   Test: {len(self.test_df)} samples")
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def get_features_and_labels(self, df):
        """Combine embedding and similarity features"""
        emb_cols = [col for col in df.columns if col.startswith('emb_')]
        sim_cols = ['similarity_to_real', 'similarity_to_fake']
        feature_cols = emb_cols + [c for c in sim_cols if c in df.columns]

        X = df[feature_cols].values
        y = df['label'].values
        return X, y, feature_cols

    def train_models(self):
        """Train Logistic Regression and Random Forest on combined features"""
        print("\n🔧 Training Baseline D models...")
        X_train, y_train, feature_cols = self.get_features_and_labels(self.train_df)

        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Logistic Regression
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train_scaled, y_train)

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        self.models = {'Logistic Regression': logreg, 'Random Forest': rf}
        print(f"✅ Models trained on {len(feature_cols)} features (embeddings + similarities)")
        return self.models, feature_cols

    def evaluate_model(self, model_name, model, df, split_name):
        """Evaluate model on dataset"""
        print(f"\n🎯 Evaluating {model_name} on {split_name.upper()} set...")
        X, y_true, feature_cols = self.get_features_and_labels(df)

        if model_name == 'Logistic Regression':
            X = self.scaler.transform(X)

        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        print(f"📈 Accuracy: {acc:.4f}")
        print(f"\n📊 Confusion Matrix:\n{cm}")
        print(f"\n📋 Classification Report:\n{classification_report(y_true, y_pred)}")

        return acc, cm, y_pred

    def save_results(self, results_summary, output_path='baselineD_results.csv'):
        """Save summary results"""
        results_df = pd.DataFrame(results_summary)
        results_df.to_csv(output_path, index=False)
        print(f"💾 Results saved to {output_path}")
        return results_df

    def visualize_results(self, results_summary):
        """Plot accuracy comparison across models and splits"""
        print("\n📊 Creating Baseline D visualizations...")

        df = pd.DataFrame(results_summary)
        fig, ax = plt.subplots(figsize=(8,6))

        for model_name in df['model'].unique():
            model_df = df[df['model'] == model_name]
            ax.plot(model_df['split'], model_df['accuracy'], marker='o', label=model_name)

        ax.set_title('Baseline D - Accuracy Across Splits (Embeddings + Similarities)')
        ax.set_xlabel('Data Split')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('baselineD_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_baselineD_analysis(self):
        """Full Baseline D pipeline"""
        print("🚀 RUNNING BASELINE D: COMBINED EMBEDDINGS + SIMILARITIES")
        print("=" * 60)

        if not self.load_split_data():
            print("❌ Failed to load data. Exiting.")
            return None

        models, feature_cols = self.train_models()
        all_results = []

        for model_name, model in models.items():
            for split_name, df in [('train', self.train_df), ('validation', self.val_df), ('test', self.test_df)]:
                acc, cm, preds = self.evaluate_model(model_name, model, df, split_name)
                all_results.append({
                    'model': model_name,
                    'split': split_name,
                    'accuracy': acc,
                    'samples': len(df),
                    'algorithm': model_name,
                    'features_used': 'embeddings+similarities'
                })

                # Save detailed predictions
                X, y_true, feature_cols = self.get_features_and_labels(df)
                detailed_df = df[feature_cols + ['label']].copy()
                detailed_df['predicted'] = preds
                detailed_df.to_csv(f'baselineD_{model_name.replace(" ", "_").lower()}_{split_name}_predictions.csv', index=False)

        results_df = self.save_results(all_results)
        self.visualize_results(all_results)

        print("\n✅ BASELINE D COMPLETED SUCCESSFULLY!")
        print("📁 Saved files:")
        print("   - baselineD_results.csv")
        print("   - baselineD_results.png")
        print("   - baselineD_[model]_[split]_predictions.csv")
        return results_df


def main():
    print("🎯 BASELINE D: COMBINED EMBEDDINGS + COSINE SIMILARITIES")
    print("=========================================================")
    print("This baseline trains Logistic Regression and Random Forest")
    print("models using both embeddings and cosine similarity features.")
    print("=========================================================")

    baselineD = BaselineDModel()
    try:
        baselineD.run_baselineD_analysis()
    except Exception as e:
        print(f"💥 Error running Baseline D: {e}")


if __name__ == "__main__":
    main()
