# [file name]: baselineB_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class BaselineBModel:
    def __init__(self):
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.model_type = "Trained Classifier on Embeddings"
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
        """Extract embedding features and labels"""
        feature_cols = [col for col in df.columns if col.startswith('emb_')]
        X = df[feature_cols].values
        y = df['label'].values
        return X, y

    def train_models(self):
        """Train Logistic Regression and Random Forest models"""
        print("\n🔧 Training Baseline B models...")
        X_train, y_train = self.get_features_and_labels(self.train_df)

        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Logistic Regression
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train_scaled, y_train)

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        self.models = {'Logistic Regression': logreg, 'Random Forest': rf}

        print("✅ Models trained successfully!")
        return self.models

    def evaluate_model(self, model_name, model, df, split_name):
        """Evaluate a single model on a given dataset"""
        print(f"\n🎯 Evaluating {model_name} on {split_name.upper()} set...")
        X, y_true = self.get_features_and_labels(df)

        if model_name == 'Logistic Regression':
            X = self.scaler.transform(X)

        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        print(f"📈 Accuracy: {acc:.4f}")
        print(f"\n📊 Confusion Matrix:\n{cm}")
        print(f"\n📋 Classification Report:\n{classification_report(y_true, y_pred)}")

        return acc, cm, y_pred

    def save_results(self, results_summary, output_path='baselineB_results.csv'):
        """Save Baseline B performance results"""
        results_df = pd.DataFrame(results_summary)
        results_df.to_csv(output_path, index=False)
        print(f"💾 Results saved to {output_path}")
        return results_df

    def visualize_results(self, results_summary):
        """Visualize accuracy across models and splits"""
        print("\n📊 Creating Baseline B visualizations...")

        df = pd.DataFrame(results_summary)
        fig, ax = plt.subplots(figsize=(8,6))

        for model_name in df['model'].unique():
            model_df = df[df['model'] == model_name]
            ax.plot(model_df['split'], model_df['accuracy'], marker='o', label=model_name)

        ax.set_title('Baseline B - Accuracy Across Splits')
        ax.set_xlabel('Data Split')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('baselineB_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_baselineB_analysis(self):
        """Full Baseline B pipeline"""
        print("🚀 RUNNING BASELINE B: TRAINED MODEL ON EMBEDDINGS")
        print("=" * 60)

        if not self.load_split_data():
            print("❌ Failed to load data. Exiting.")
            return None

        models = self.train_models()
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
                    'features_used': 'embedding_features_only'
                })

                # Save detailed predictions
                feature_cols = [col for col in df.columns if col.startswith('emb_')]
                detailed_df = df[feature_cols + ['label']].copy()
                detailed_df['predicted'] = preds
                detailed_df.to_csv(f'baselineB_{model_name.replace(" ", "_").lower()}_{split_name}_predictions.csv', index=False)

        results_df = self.save_results(all_results)
        self.visualize_results(all_results)

        print("\n✅ BASELINE B COMPLETED SUCCESSFULLY!")
        print("📁 Saved files:")
        print("   - baselineB_results.csv")
        print("   - baselineB_results.png")
        print("   - baselineB_[model]_[split]_predictions.csv")
        return results_df


def main():
    print("🎯 BASELINE B: TRAINED CLASSIFIER ON EMBEDDINGS")
    print("=================================================")
    print("This baseline trains Logistic Regression and Random Forest")
    print("models on raw document embeddings.")
    print("=================================================")

    baselineB = BaselineBModel()
    try:
        baselineB.run_baselineB_analysis()
    except Exception as e:
        print(f"💥 Error running Baseline B: {e}")


if __name__ == "__main__":
    main()
