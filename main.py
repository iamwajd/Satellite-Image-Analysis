import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Subset

import numpy as np
import matplotlib.pyplot as plt
import umap

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class SatelliteAnalyzer:
    def __init__(self, data_dir, batch_size=32, device=None):
        # Set device to GPU if available, otherwise CPU
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Define preprocessing transforms compatible with ResNet18
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.model = self._setup_model()

    def _setup_model(self):
        # Load pre-trained ResNet18 and remove final classification layer
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        feature_extractor = nn.Sequential(*(list(base_model.children())[:-1]))
        return feature_extractor.to(self.device).eval()

    def extract_features(self, sample_size=1200):
        # Load dataset and randomly select a subset of images
        dataset = ImageFolder(self.data_dir, transform=self.transform)
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)

        features, labels = [], []
        print("[*] Extracting CNN features...")
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                outputs = outputs.view(outputs.size(0), -1)  # Flatten feature maps
                features.append(outputs.cpu().numpy())
                labels.extend(targets.numpy())

        # Return features, labels, full dataset reference, and sampled indices
        return np.concatenate(features), np.array(labels), dataset, indices

    def train_classifier(self, features, labels):
        # Split features into train/test sets and train a linear SVM
        print("[*] Training SVM classifier...")
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        clf = SVC(kernel='linear', C=1.0)
        clf.fit(X_train, y_train)

        # Evaluate classifier accuracy on the test set
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n[!] Model Accuracy: {acc:.4f}")
        return clf, X_test, y_test, preds

    def plot_confusion(self, y_test, preds, class_names):
        # Generate and display a confusion matrix to analyze model performance
        print("[*] Displaying Confusion Matrix...")
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(xticks_rotation=45, cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix")
        plt.show()

    def visualize_umap(self, features, labels, dataset, indices, num_thumbnails=30):
        # Apply UMAP for dimensionality reduction to 2D for visualization
        print("[*] Running UMAP dimensionality reduction...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(features)

        # Scatter plot of UMAP embedding colored by class labels
        fig, ax = plt.subplots(figsize=(15, 10))
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap='Spectral',
            s=20,
            alpha=0.6
        )

        # Add legend mapping colors to class names
        ax.legend(handles=scatter.legend_elements()[0], labels=dataset.classes,
                  title="Classes", loc="upper left", bbox_to_anchor=(1, 1))

        # Overlay thumbnails of sample images on top of scatter points
        step = max(1, len(embedding) // num_thumbnails)
        for i in range(0, len(embedding), step):
            img_path, _ = dataset.samples[indices[i]]
            img = plt.imread(img_path)
            imagebox = OffsetImage(img, zoom=0.5)
            ab = AnnotationBbox(imagebox, (embedding[i, 0], embedding[i, 1]), frameon=False)
            ax.add_artist(ab)

        plt.title("Satellite Image Clustering (ResNet18 + UMAP)")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    analyzer = SatelliteAnalyzer(data_dir="./data")
    
    # Step 1: Extract CNN features from a subset of satellite images
    features, labels, dataset, indices = analyzer.extract_features(sample_size=1200)
    
    # Step 2: Train SVM classifier and evaluate performance
    clf, X_test, y_test, preds = analyzer.train_classifier(features, labels)
    
    # Step 3: Show confusion matrix for classification results
    analyzer.plot_confusion(y_test, preds, dataset.classes)
    
    # Step 4: Visualize high-dimensional features in 2D using UMAP with sample thumbnails
    analyzer.visualize_umap(features, labels, dataset, indices)