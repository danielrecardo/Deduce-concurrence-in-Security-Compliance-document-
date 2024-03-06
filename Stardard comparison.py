import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_path):
    return pd.read_excel(file_path)


def preprocess_text(text):
    # Check if the value is NaN, and if so, return an empty string
    if isinstance(text, float) and math.isnan(text):
        return ''

    # Otherwise, apply the text preprocessing steps
    return str(text).lower()


def calculate_cosine_similarity(matrix):
    return cosine_similarity(matrix)

def main():
    # Load data from Excel files
    nist_data = load_data(r"C:\Users\aditt\Documents\nist_800_53.xlsx")
    iso_data = load_data(r"C:\Users\aditt\Documents\iso_27001_2013.xlsx")

    # Preprocess control descriptions
    nist_data['Control_Description'] = nist_data['Control_Description'].apply(preprocess_text)
    iso_data['Control_Description'] = iso_data['Control_Description'].apply(preprocess_text)

    # Combine control descriptions from both standards
    all_descriptions = nist_data['Control_Description'].tolist() + iso_data['Control_Description'].tolist()

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_descriptions)

    # Calculate cosine similarity
    similarity_matrix = calculate_cosine_similarity(tfidf_matrix)

    # Extract similar controls
    nist_indices, iso_indices = similarity_matrix[:len(nist_data), len(nist_data):].nonzero()

    # Create a mapping DataFrame
    mapping_df = pd.DataFrame({
        'NIST_Control': nist_data.iloc[nist_indices]['Control_Number'].tolist(),
        'ISO_Control': iso_data.iloc[iso_indices]['Control_Number'].tolist(),
        'Similarity_Score': similarity_matrix[nist_indices, iso_indices].tolist()
    })

    # Save the mapping DataFrame to an output Excel file
    output_file_path = r"C:\Users\aditt\Documents\control_mapping_output.xlsx"
    mapping_df.to_excel(output_file_path, index=False)


if __name__ == "__main__":
    main()
