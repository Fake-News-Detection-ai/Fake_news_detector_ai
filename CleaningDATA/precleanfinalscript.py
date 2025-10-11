import pandas as pd
import re
from tqdm import tqdm  # for the loading bar

# 1️⃣ Load the dataset
df = pd.read_csv("fake_news_dataset.csv")

# 2️⃣ Remove duplicates and missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# 3️⃣ Define the cleaning function
def clean_text(text):
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

    return text

# 4️⃣ Add progress bar to Pandas apply()
tqdm.pandas(desc="🧹 Cleaning text")

# 5️⃣ Apply cleaning to all text columns with a progress bar
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"\nProcessing column: {col}")
        df[col] = df[col].progress_apply(clean_text)

# 6️⃣ Save the cleaned dataset
df.to_csv("precleaned_data.csv", index=False)

print("\n✅ Data cleaned successfully and saved as precleaned_data.csv")

