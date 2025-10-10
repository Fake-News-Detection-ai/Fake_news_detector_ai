from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to remove similar articles based on cosine similarity
def remove_similar_articles(vectors, threshold=0.9):
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(vectors)
    
    # Keep track of indices to remove
    to_remove = set()
    
    # Loop through the matrix and mark similar articles
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i][j] > threshold:
                to_remove.add(j)  # remove the later article to keep the first
    
    # Remove duplicates
    filtered_vectors = np.delete(vectors, list(to_remove), axis=0)
    
    return filtered_vectors, to_remove

# Example usage (replace 'vectors' with your friend's vectorized dataset)
# filtered_vectors, removed_indices = remove_similar_articles(vectors)
# print("Removed duplicate indices:", removed_indices)
# print("Filtered vectors shape:", filtered_vectors.shape)
