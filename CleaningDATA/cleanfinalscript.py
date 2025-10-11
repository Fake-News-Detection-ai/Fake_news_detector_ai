import pandas as pd
import spacy
from tqdm import tqdm  # for the loading bar

#Load dataset using pandas
data = pd.read_csv("precleaned_data.csv")

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    # Process the text with spaCy
    doc = nlp(text)
    
    # Lemmatize, remove stop words
    cleaned_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    
    # Join tokens back into a single string from ['fake' , 'text'] to 'fake text'
    cleaned_text = ' '.join(cleaned_tokens)
    
    return cleaned_text

# Add progress bar to Pandas apply()
tqdm.pandas(desc="🧹 Cleaning text")

print("Starting cleaning process...")

# Apply the cleaning function to all text column for each row
for col in data.columns:
    if data[col].dtype == 'object':
        print(f"\nProcessing column: {col}")
        data[col] = data[col].progress_apply(clean_text)

# Save the cleaned data
data.to_csv("cleaned_dataset.csv", index=False)

print("Data cleaned successfully and saved as cleaned_data.csv")
