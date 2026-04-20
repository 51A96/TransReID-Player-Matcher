import os
import faiss
import numpy as np
from transreid import TransReID

def process_images(training_dir:str):
    # Load the TransReID model
    model = TransReID(pretrained=True)
    feature_list = []
    metadata_list = []

    # Iterate through all images in the training directory
    for img_name in os.listdir(training_dir):
        img_path = os.path.join(training_dir, img_name)
        if img_path.endswith(('.png', '.jpg', '.jpeg')):
            # Extract features
            features = model.extract_features(img_path)
            feature_list.append(features)
            metadata_list.append({'image_name': img_name, 'path': img_path})

    return np.array(feature_list), metadata_list


def create_faiss_index(features:np.ndarray):
    # Create a FAISS index
    dim = features.shape[1]  # Feature dimension
    index = faiss.IndexFlatL2(dim)
    index.add(features)
    return index


def save_features_and_metadata(features:np.ndarray, metadata_list:list, output_dir:str):
    os.makedirs(output_dir, exist_ok=True)
    # Save features
    np.save(os.path.join(output_dir, 'features.npy'), features)
    # Save metadata
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata_list, f)


def main(training_dir:str, output_dir:str):
    features, metadata = process_images(training_dir)
    index = create_faiss_index(features)
    save_features_and_metadata(features, metadata, output_dir)


if __name__ == '__main__':
    # Example usage
    training_directory = 'path/to/training_directory'
    output_directory = 'path/to/output_directory'
    main(training_directory, output_directory)