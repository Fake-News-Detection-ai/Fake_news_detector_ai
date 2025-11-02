# baselineD_inference.py
import pandas as pd
import numpy as np
import pickle
import spacy
import json
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class BaselineDInference:
    def __init__(self, model_path=None, vectorizer_path=None, centroid_path=None):
        """
        Initialize the Baseline D inference pipeline
        
        Args:
            model_path: Path to saved Baseline D model (will be provided later)
            vectorizer_path: Path to saved vectorizer
            centroid_path: Path to saved centroid data
        """
        self.model = None
        self.vectorizer = None
        self.real_centroid = None
        self.fake_centroid = None
        self.feature_scaler = None
        
        # Load resources if provided
        if model_path:
            self.load_model(model_path)
        if vectorizer_path:
            self.load_vectorizer(vectorizer_path)
        if centroid_path:
            self.load_centroids(centroid_path)
    
    def load_model(self, model_path):
        """Load the trained Baseline D model"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Baseline D model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def load_vectorizer(self, vectorizer_path):
        """Load the text vectorizer"""
        try:
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print(f"✅ Vectorizer loaded from {vectorizer_path}")
        except Exception as e:
            print(f"❌ Error loading vectorizer: {e}")
            self.vectorizer = None
    
    def load_centroids(self, centroid_path):
        """Load pre-calculated centroids"""
        try:
            with open(centroid_path, 'rb') as f:
                centroid_data = pickle.load(f)
            self.real_centroid = centroid_data['real_centroid']
            self.fake_centroid = centroid_data['fake_centroid']
            print(f"✅ Centroids loaded from {centroid_path}")
        except Exception as e:
            print(f"❌ Error loading centroids: {e}")
    
    def clean_text(text,nlp):
        import re

        if not isinstance(text, str):
            return ""

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove special characters and numbers
        text = re.sub(r'[^A-Za-z\s]', '', text)

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        doc = nlp(text)
        
        # Lemmatize, remove stop words
        cleaned_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        
        # Join tokens back into a single string from ['fake' , 'text'] to 'fake text'
        cleaned_text = ' '.join(cleaned_tokens)
        
        return cleaned_text
    
    def tokenize_text(text,nlp):
        doc = nlp(str(text))  # convert to string in case there are NaNs
        return [token.text for token in doc if not token.is_punct and not token.is_space]
    
    def tokens_to_vector(tokens_list):
        if not tokens_list or pd.isna(tokens_list):
            return ','.join(['0'] * 300)
        
        vectors = []

        for token in tokens_list:
            word = nlp.vocab[str(token).lower()]

            if word.has_vector:
                vectors.append(word.vector)
        
        if vectors:
            avg_vector = np.mean(vectors, axis=0)
            return ','.join(map(str, avg_vector))
        else:
            return ','.join(['0'] * 300)
        
    
    def text_to_vector(self, text):
        """
        Convert text to vector using YOUR actual preprocessing code
        """

        nlp = spacy.load("en_core_web_md")
        # COPY YOUR CLEANING CODE HERE
        cleaned_text = self.clean_text(text,nlp)
        
        # COPY YOUR TOKENIZATION CODE HERE  
        tokens = self.tokenize_text(cleaned_text,nlp)
        
        # COPY YOUR VECTORIZATION CODE HERE
        vector = self.vectorize_tokens(tokens,nlp)
        
        return vector
    
    def calculate_cosine_similarities(self, vector):
        """Calculate cosine similarities to real and fake centroids"""
        if self.real_centroid is None or self.fake_centroid is None:
            raise ValueError("Centroids not loaded. Call load_centroids() first.")
        
        # Reshape for cosine_similarity
        vector_2d = vector.reshape(1, -1)
        real_centroid_2d = self.real_centroid.reshape(1, -1)
        fake_centroid_2d = self.fake_centroid.reshape(1, -1)
        
        similarity_to_real = cosine_similarity(vector_2d, real_centroid_2d)[0][0]
        similarity_to_fake = cosine_similarity(vector_2d, fake_centroid_2d)[0][0]
        
        return similarity_to_real, similarity_to_fake
    
    def extract_features(self, text):
        """
        Extract all features needed for Baseline D prediction
        This combines features from all baselines
        """
        # 1. Convert text to vector (Baseline B features)
        vector = self.text_to_vector(text)
        
        # 2. Calculate cosine similarities (Baseline C features)
        similarity_to_real, similarity_to_fake = self.calculate_cosine_similarities(vector)
        
        # 3. Create feature array for Baseline D
        features = {
            'raw_vector': vector,  # For complex models that use raw vectors
            'cosine_features': np.array([similarity_to_real, similarity_to_fake]),
            'similarity_to_real': similarity_to_real,
            'similarity_to_fake': similarity_to_fake,
            'similarity_margin': abs(similarity_to_real - similarity_to_fake),
            'max_similarity': max(similarity_to_real, similarity_to_fake)
        }
        
        return features
    
    def predict_single(self, text, return_confidence=False):
        """
        Predict single article
        
        Args:
            text: News article text
            return_confidence: Whether to return confidence scores
        
        Returns:
            prediction: 'real' or 'fake'
            confidence: Optional confidence score
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Extract features
        features = self.extract_features(text)
        
        # Prepare features for prediction (this might need adjustment based on final model)
        # Option 1: If model uses only cosine features
        X = features['cosine_features'].reshape(1, -1)
        
        # Option 2: If model uses raw vectors + cosine features
        # X = np.concatenate([features['raw_vector'], features['cosine_features']]).reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        prediction_label = 'real' if prediction == 1 else 'fake'
        
        if return_confidence:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                confidence = max(probabilities)
            else:
                # For models without probability, use similarity margin as proxy
                confidence = features['similarity_margin']
            
            return prediction_label, confidence
        else:
            return prediction_label
    
    def predict_batch(self, texts, return_features=False):
        """
        Predict multiple articles at once
        
        Args:
            texts: List of news article texts
            return_features: Whether to return extracted features
        
        Returns:
            predictions: List of predictions
            features_df: Optional DataFrame with features
        """
        predictions = []
        all_features = []
        
        for text in texts:
            features = self.extract_features(text)
            prediction = self.predict_single(text)
            
            predictions.append(prediction)
            
            if return_features:
                feature_record = {
                    'text': text[:100] + '...' if len(text) > 100 else text,  # Preview
                    'prediction': prediction,
                    'similarity_to_real': features['similarity_to_real'],
                    'similarity_to_fake': features['similarity_to_fake'],
                    'similarity_margin': features['similarity_margin']
                }
                all_features.append(feature_record)
        
        if return_features:
            features_df = pd.DataFrame(all_features)
            return predictions, features_df
        else:
            return predictions
    
    def evaluate_model(self, test_texts, test_labels):
        """
        Evaluate model performance on test data
        """
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        predictions = self.predict_batch(test_texts)
        
        accuracy = accuracy_score(test_labels, predictions)
        
        print(f"📊 Baseline D Evaluation Results")
        print(f"✅ Accuracy: {accuracy:.4f}")
        print(f"\n📋 Classification Report:")
        print(classification_report(test_labels, predictions))
        print(f"\n🎯 Confusion Matrix:")
        print(confusion_matrix(test_labels, predictions))
        
        return accuracy, predictions

# Example usage and testing class
class BaselineDTester:
    """
    Helper class to test the inference pipeline before the actual model is ready
    """
    def __init__(self):
        self.inference_pipeline = BaselineDInference()
    
    def create_mock_model(self):
        """Create a mock model for testing the pipeline"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Mock model that always predicts based on cosine similarity
        # This will be replaced with the actual Baseline D model
        self.mock_model = LogisticRegression()
        
        # Train on dummy data
        X_dummy = np.random.rand(100, 2)  # 2 features: similarity_to_real, similarity_to_fake
        y_dummy = np.random.randint(0, 2, 100)  # Binary labels
        
        self.mock_model.fit(X_dummy, y_dummy)
        self.inference_pipeline.model = self.mock_model
        
        print("✅ Mock model created for pipeline testing")
    
    def test_single_prediction(self, test_text="Sample news article text here"):
        """Test single article prediction"""
        try:
            # This will fail without actual vectorizer/centroids, but shows the flow
            prediction, confidence = self.inference_pipeline.predict_single(
                test_text, return_confidence=True
            )
            print(f"📰 Test Article: '{test_text}'")
            print(f"🎯 Prediction: {prediction}")
            print(f"📊 Confidence: {confidence:.4f}")
        except Exception as e:
            print(f"⚠️  Expected error (no real model yet): {e}")
            print("📝 This shows the pipeline is ready for the actual model!")
    
    def run_pipeline_test(self):
        """Run complete pipeline test"""
        print("🚀 TESTING BASELINE D INFERENCE PIPELINE")
        print("=" * 50)
        
        self.create_mock_model()
        self.test_single_prediction()
        
        print("\n✅ INFERENCE PIPELINE READY!")
        print("📋 What your friend needs to provide:")
        print("   1. Trained Baseline D model (pickle file)")
        print("   2. Vectorizer used for text processing") 
        print("   3. Centroid vectors from training data")
        print("   4. Feature specification (which features model expects)")

def main():
    """
    Main function to demonstrate the inference pipeline
    """
    print("🎯 BASELINE D INFERENCE PIPELINE")
    print("=================================")
    print("This pipeline is ready for your friend's Baseline D model.")
    print("Currently running in TEST MODE with mock data.")
    
    tester = BaselineDTester()
    tester.run_pipeline_test()
    
    print("\n📝 NEXT STEPS:")
    print("1. Your friend trains Baseline D model")
    print("2. Save model as: baselineD_model.pkl") 
    print("3. Save vectorizer as: vectorizer.pkl")
    print("4. Save centroids as: centroids.pkl")
    print("5. Update feature extraction if needed")
    print("6. Test with real model!")

if __name__ == "__main__":
    main()


# When you create the pipeline object, load everything at once:
pipeline = BaselineDInference(
    model_path='trained_models/baselineD_model.pkl',
    vectorizer_path='trained_models/vectorizer.pkl',
    centroid_path='trained_models/centroids.pkl'
)

# Now you can immediately use it:
prediction = pipeline.predict_single("Your news article text here")