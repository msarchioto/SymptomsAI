import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def main():
    # 1. Load the dataset
    data = pd.read_csv('dataset/symptoms_dataset_spaces.csv')

    # 2. Preprocess the dataset
    # Combine all symptom columns into one string per disease
    data['Symptoms'] = data[
            [
                'Symptom_1',
                'Symptom_2',
                'Symptom_3',
                'Symptom_4',
                'Symptom_5',
                'Symptom_6',
                'Symptom_7',
                'Symptom_8',
                'Symptom_9',
                'Symptom_10',
                'Symptom_11',
                'Symptom_12',
                'Symptom_13',
                'Symptom_14',
                'Symptom_15',
                'Symptom_16',
                'Symptom_17'
            ]
        ].apply(lambda row: ', '.join(row.dropna().astype(str)), axis=1)

    # 3. Initialize the embedding model
    # all-MiniLM-L6-v2 model is a lightweight, efficient, and versatile option that
    # balances accuracy with speed, making it ideal for embedding in a local,
    # resource-constrained environment
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 4. Generate embeddings for each disease-symptom pair
    # Create a new column 'Embedding' containing the vector for each disease-symptom pair
    data['Embedding'] = data.apply(lambda row: model.encode(f"{row['Disease']}: {row['Symptoms']}"), axis=1)

    # 5. Prepare data for FAISS
    # Convert embeddings into a numpy array
    embeddings = np.vstack(data['Embedding'].values)

    # Create unique IDs for each entry (optional, for easier tracking)
    ids = np.array(range(len(data)))

    # 6. Create a FAISS index (L2 similarity)
    dimension = embeddings.shape[1]  # Size of each embedding vector
    index = faiss.IndexFlatL2(dimension)

    # Optionally, use IndexIDMap for custom IDs
    index_with_ids = faiss.IndexIDMap(index)
    index_with_ids.add_with_ids(embeddings, ids)

    # 7. Save the FAISS index to disk
    faiss.write_index(index_with_ids, "vector_db/disease_symptoms_index.faiss")

    print("FAISS index saved successfully.")

if __name__ == '__main__':
    main()
