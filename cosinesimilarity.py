import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the spaCy model (make sure it's downloaded, e.g., en_core_web_md)
nlp = spacy.load("en_core_web_md")

# Example dataset (in your case, this would be the cleaned and preprocessed news dataset)
texts = [
    "The president gave a speech about the economy today.",
    "Today, the president spoke about the country's economy.",
    "A new species of bird was found in the Amazon rainforest.",
    "The economy is improving according to the president's speech."
]

# Convert texts to spaCy Doc objects
docs = [nlp(text) for text in texts]

# Extract the document vectors
vectors = np.array([doc.vector for doc in docs])

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(vectors)

# Display the similarity matrix
print("Cosine Similarity Matrix:")
print(similarity_matrix)

# Identify highly similar pairs
threshold = 0.9
print("\nHighly similar pairs (similarity > 0.9):")
for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):
        if similarity_matrix[i][j] > threshold:
            print("Article", i, "and Article", j, "are similar with score:", similarity_matrix[i][j])
