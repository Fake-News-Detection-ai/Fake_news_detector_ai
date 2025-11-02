import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Convert the list of tokens to average vector
def tokens_to_vector(tokens_list):
    # Checking for empty or invalid data
    if not tokens_list or pd.isna(tokens_list):
        return ','.join(['0'] * 300)
    
    # Get vectors for all tokens that have vectors
    vectors = []

    # loops through each token in a row(tokens_list)
    for token in tokens_list:
        # assigns each word a vector by looking up in the spaCy dictionary
        # spaCy saves mathematical representations
        # maps the word as a list of 300 numbers which capture statistical patterns and relationships between words
        # e.g: capture relaitonship between tax and government
        word = nlp.vocab[str(token).lower()]

        # Checks if this word has a pre-trained vector
        # Some rare words might not have vectors
        if word.has_vector:
            vectors.append(word.vector)
    
    # Return average vector or zeros if no vectors found
    if vectors:
        avg_vector = np.mean(vectors, axis=0)
        # Convert directly to comma-separated string
        return ','.join(map(str, avg_vector))
    else:
        return ','.join(['0'] * 300)
    
    
# Read the tokenized CSV
df = pd.read_csv("tokenized_dataset.csv")

# Convert tokens to vectors with progress bar
print("🔄 Converting tokens to vectors...")
tqdm.pandas(desc="Processing")
df['vectors'] = df['tokens'].progress_apply(tokens_to_vector)

# Save the vectorized CSV
df.to_csv('vectorized_data.csv', index=False)

print("✅ Successfully converted tokens to vectors!")
print(f"Processed {len(df)} rows")
print("Saved as 'vectorized_data.csv' for similarity calculation")