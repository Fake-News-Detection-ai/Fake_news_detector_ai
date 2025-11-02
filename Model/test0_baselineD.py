# use_baselineD.py
from baselineD_inference import BaselineDInference  # FIXED: Added missing 'e'

def main():
    print("🚀 TESTING BASELINE D PIPELINE")
    print("=" * 40)
    
    # First, let's test if the pipeline structure works
    # We'll use the mock tester since you don't have the actual model files yet
    from baselineD_inference import BaselineDTester
    
    tester = BaselineDTester()
    tester.run_pipeline_test()
    
    print("\n" + "=" * 40)
    print("📝 TO USE WITH ACTUAL MODEL FILES:")
    print("=" * 40)
    print("When your friend provides the files, use this code:")
    print("""
    pipeline = BaselineDInference(
        model_path='trained_models/baselineD_model.pkl',
        centroid_path='trained_models/centroids.pkl'
    )
    
    test_article = "Your news article text here"
    prediction, confidence = pipeline.predict_single(test_article, return_confidence=True)
    print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
    """)

if __name__ == "__main__":
    main()