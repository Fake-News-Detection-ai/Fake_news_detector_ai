from flask import Flask, render_template, request, jsonify
import pickle
import os
import re

class FakeNewsPredictor:
    def __init__(self, model_path='enhanced_fake_news_model_v2.pkl'):
        """Load the saved model and vectorizer"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Please place it in the same directory as app.py")
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
        
        print(f"✓ Model loaded from {model_path}")
    
    @staticmethod
    def _clean_for_ml(text):
        """Clean text for ML model (same as training)"""
        t = str(text).lower()
        t = re.sub(r'[^a-zA-Z\s]', ' ', t)
        return ' '.join(t.split())
    
    def predict(self, news_text):
        """
        Make prediction using the loaded model
        Returns: prediction (0=FAKE, 1=TRUE), confidence, probabilities
        """
        # Clean the text
        cleaned = self._clean_for_ml(news_text)
        
        # Transform using vectorizer
        vec = self.vectorizer.transform([cleaned])
        
        # Get prediction and probabilities
        prediction = int(self.model.predict(vec)[0])
        probabilities = self.model.predict_proba(vec)[0]
        
        # Calculate confidence
        fake_prob = float(probabilities[0])
        true_prob = float(probabilities[1])
        confidence = max(fake_prob, true_prob)
        
        return {
            'prediction': prediction,  # 0=FAKE, 1=TRUE
            'confidence': confidence,
            'fake_probability': fake_prob,
            'true_probability': true_prob,
            'label': 'TRUE' if prediction == 1 else 'FAKE'
        }

# Initialize Flask app
app = Flask(__name__)

# Load the model
try:
    predictor = FakeNewsPredictor('enhanced_fake_news_model_v2.pkl')
except FileNotFoundError as e:
    print(f"\n⚠️  ERROR: {e}")
    print("Please place 'enhanced_fake_news_model_v2.pkl' in the same directory as app.py\n")
    exit(1)

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        news_text = data.get('text', '').strip()
        
        if not news_text:
            return jsonify({'error': 'Please enter some text to analyze'}), 400
        
        if len(news_text) < 20:
            return jsonify({'error': 'Text is too short. Please enter a meaningful news article or claim.'}), 400
        
        # Make prediction
        result = predictor.predict(news_text)
        
        # Return results
        return jsonify({
            'success': True,
            'prediction': result['label'],
            'confidence': round(result['confidence'] * 100, 2),
            'fake_probability': round(result['fake_probability'] * 100, 2),
            'true_probability': round(result['true_probability'] * 100, 2),
            'analysis': {
                'text_length': len(news_text),
                'word_count': len(news_text.split())
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 FAKE NEWS DETECTOR - WEB APP")
    print("="*60)
    print("✓ Model loaded successfully")
    print("✓ Starting server...")
    print("\n👉 Open your browser and go to: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)
