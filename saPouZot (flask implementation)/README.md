# 🤖 AI Fake News Detector - Web Interface

A modern, beautiful web application for detecting fake news using Machine Learning and Web Verification!

## ✨ Features

- **ML Model Analysis**: Uses your trained LogisticRegression model with TF-IDF vectorization
- **Web Verification**: Searches trusted sources and analyzes sentiment
- **Beautiful UI**: Modern gradient design with smooth animations
- **Confidence Scores**: Visual probability bars showing fake vs. real likelihood
- **Similar Articles**: Displays related articles with sentiment analysis (support/debunk/neutral)
- **Real-time Analysis**: Instant feedback with loading animations

## 📁 Project Structure

```
fake-news-detector/
├── app.py                              # Flask backend
├── templates/
│   └── index.html                      # Frontend UI
├── requirements.txt                    # Python dependencies
├── enhanced_fake_news_detector_v2.py   # Your training model class
└── enhanced_fake_news_model_v2.pkl     # Your trained model (YOU NEED TO ADD THIS!)
```

## 🚀 Setup Instructions

### 1. Install Dependencies

Open your terminal/command prompt and run:

```bash
pip install -r requirements.txt
```

### 2. Copy Your Model Files

**IMPORTANT:** You need to place these files in the same directory as `app.py`:

- `enhanced_fake_news_model_v2.pkl` - Your trained model file
- `enhanced_fake_news_detector_v2.py` - Your detector class (already included)

### 3. Run the Application

```bash
python app.py
```

You should see:
```
🤖 AI FAKE NEWS DETECTOR - Starting Server
============================================================
Model loaded from enhanced_fake_news_model_v2.pkl

✅ Server ready!
📱 Open http://127.0.0.1:5000 in your browser
============================================================
```

### 4. Open in Browser

Navigate to: **http://127.0.0.1:5000**

## 🎥 Demo Guide

### For Your Presentation:

1. **Prepare Example Articles**:
   - Have 2-3 fake news examples ready (copy from your dataset or find online)
   - Have 2-3 real news examples ready
   
2. **What to Show**:
   - Paste a FAKE news article → Show the red "LIKELY FAKE NEWS" verdict
   - Show the ML probabilities bar (should show high P(FAKE))
   - Point out the web verification results and similar articles
   - Paste a REAL news article → Show the green "LIKELY REAL NEWS" verdict
   - Show how confidence scores work
   - Highlight the trusted sources found

3. **Cool Features to Mention**:
   - "Our model uses TF-IDF with 50,000 features and bigrams"
   - "We calibrate our LogisticRegression for better probability estimates"
   - "The system cross-references with trusted sources like Reuters, BBC, CNN, etc."
   - "We analyze sentiment: supporting, debunking, or neutral articles"
   - "Relevance filtering ensures only related articles influence the verdict"

## 🎨 UI Features

- **Animated gradient background** - Purple to violet
- **Floating particles** - For that modern tech vibe
- **Smooth transitions** - Professional animations
- **Color-coded verdicts**:
  - 🟢 Green = Real News
  - 🔴 Red = Fake News
  - ⚪ Gray = Not a news claim
- **Progress bars** - Visual probability indicators
- **Article cards** - Beautiful display of similar articles with sentiment badges

## 📊 Example Inputs

### Fake News Example:
```
Scientists discover a new drug called "TriRest" that allows humans to sleep for 72 hours straight without any side effects. The FDA has not approved this medication.
```

### Real News Example:
```
The Federal Reserve announced today that it will raise interest rates by 0.25% to combat inflation. This marks the third rate hike this year.
```

## 🛠️ Troubleshooting

### Error: "Model file not found"
- Make sure `enhanced_fake_news_model_v2.pkl` is in the same folder as `app.py`

### Error: "Module not found"
- Run `pip install -r requirements.txt` to install all dependencies

### Page won't load
- Check that the server is running (you should see the startup messages)
- Make sure you're accessing `http://127.0.0.1:5000` (not localhost)

### No web verification results
- This is normal if your internet connection blocks the web search
- The ML model will still work and give predictions

## 💡 Tips for Maximum Marks

1. **Explain the architecture**: ML model + Web verification hybrid approach
2. **Mention the features**: TF-IDF, bigrams, calibrated probabilities
3. **Show the UI/UX**: Modern design, animations, professional look
4. **Demonstrate accuracy**: Test with both fake and real news
5. **Highlight the extra features**: Web verification, similar articles, sentiment analysis

## 🎓 Technical Details

- **Framework**: Flask (Python)
- **ML Model**: Calibrated Logistic Regression
- **Vectorizer**: TF-IDF with ngrams (1,2) and 50K features
- **Frontend**: Pure HTML/CSS/JavaScript (no frameworks needed)
- **Design**: Gradient UI with CSS animations

---

Good luck with your demo! You've got this! 🚀✨
