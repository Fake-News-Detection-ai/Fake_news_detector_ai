import pandas as pd
import re
from tqdm import tqdm
import spacy

# Load spaCy model (light mode for faster checking)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# Enable progress bar for pandas apply
tqdm.pandas()

# 1️⃣ Load the dataset
file_path = "cleaned_dataset.csv"  # Change if needed
df = pd.read_csv(file_path)

print("\n📊 Dataset loaded successfully!")
print(f"➡️ Total rows: {len(df)}")
print(f"➡️ Columns: {list(df.columns)}\n")

# 2️⃣ Basic info
print("🔍 Checking for missing values...")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n🧹 Checking for duplicates...")
duplicate_count = df.duplicated().sum()
print(f"➡️ Total duplicates: {duplicate_count}\n")

# 3️⃣ Define a function to detect unwanted text
def check_dirty_text(text):
    """
    Returns True if text contains unwanted patterns (HTML, URLs, digits, symbols)
    or is linguistically 'dirty' (only stopwords, no alpha words, etc.).
    Returns False if the text is clean and meaningful.
    """

    # 1️⃣ Check for invalid data type
    if not isinstance(text, str) or text.strip() == "":
        return True   # dirty if not a valid string

    # 2️⃣ Regex checks for HTML, URLs, numbers, or special characters
    patterns = [
        r'<.*?>',           # HTML tags
        r'http\S+|www\S+',  # URLs
        r'\d+',             # Numbers
        r'[^A-Za-z\s]',     # Special characters
    ]
    for p in patterns:
        if re.search(p, text):
            return True     # dirty if pattern found

    # 3️⃣ Process text with spaCy
    doc = nlp(text)

    # 4️⃣ Check if text has at least one meaningful word
    alpha_tokens = [token for token in doc if token.is_alpha and not token.is_stop]

    if len(alpha_tokens) == 0:
        return True   # dirty if no valid tokens (only stopwords or symbols)

    # 5️⃣ If passed all checks, it's clean
    return False

# 4️⃣ Analyze which rows contain dirty data
print("🔎 Scanning text columns for unwanted patterns...\n")

dirty_rows = []
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"➡️ Checking column: {col}")
        mask = df[col].progress_apply(check_dirty_text)
        dirty_indices = df[mask].index.tolist()
        if dirty_indices:
            dirty_rows.extend(dirty_indices)
            print(f"⚠️ Found {len(dirty_indices)} dirty rows in '{col}'")
        else:
            print(f"✅ '{col}' looks clean!")

# 5️⃣ Display summary
dirty_rows = sorted(set(dirty_rows))
print("\n📋 Summary of issues found:")
print(f"➡️ Total rows needing cleaning: {len(dirty_rows)}")

if dirty_rows:
    print("\n🧾 Sample of rows that need cleaning:")
    print(df.loc[dirty_rows].head())
else:
    print("✅ Dataset looks clean! Nothing to fix.")

print("\n✅ Data analysis completed.")
