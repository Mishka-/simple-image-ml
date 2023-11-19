import tarfile
from urllib.request import urlretrieve
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Function to download and extract the archive
def download_and_extract(url, filename):
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url, filename)
        tar = tarfile.open(filename, 'r:gz')
        tar.extractall()
        tar.close()

# Download data
download_and_extract('https://commondatastorage.googleapis.com/books1000/notMNIST_small.tar.gz', 'notMNIST_small.tar.gz')

# Path to the data folder
data_folder = 'notMNIST_small/'

# Display some images
def display_images(folder, sample_size=5):
    folder_path = os.path.join(data_folder, folder)
    images = os.listdir(folder_path)
    
    plt.figure(figsize=(10, 2))

    for i in range(sample_size):
        img_name = random.choice(images)
        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path)

        # Display the image
        plt.subplot(1, sample_size, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    
    plt.show()

# Display some images from each class (A-J)
for label in os.listdir(data_folder):
    print(f"Sample images for class {label}:")
    display_images(label)

# Count the number of images in each class
class_counts = {label: len(os.listdir(os.path.join(data_folder, label))) for label in os.listdir(data_folder)}

# 1 Display the number of images in each class
print("Number of images in each class:")
for label, count in class_counts.items():
    print(f"{label}: {count}")

# 2 Check for class balance
is_balanced = all(count == list(class_counts.values())[0] for count in class_counts.values())
print("\nClasses are balanced:", is_balanced)

# Paths to images and their labels
images = []
labels = []

for label in os.listdir(data_folder):
    label_folder = os.path.join(data_folder, label)
    for img_name in os.listdir(label_folder):
        img_path = os.path.join(label_folder, img_name)
        images.append(img_path)
        labels.append(label)

# 3 Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.95, random_state=42)

print("Sizes of training, validation, and test sets:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# 4 Check for data overlap in the training set
intersection_train_val = set(X_train) & set(X_val)
intersection_train_test = set(X_train) & set(X_test)

print("Intersection between training and validation sets:", len(intersection_train_val))
print("Intersection between training and test sets:", len(intersection_train_test))

# Function to load images and convert them into numpy arrays
def load_images(image_paths):
    images = [np.array(Image.open(img_path).convert('L')).flatten() for img_path in image_paths]
    return np.array(images)

# Load data for training
X_train_data = load_images(X_train)
y_train_data = np.array(y_train)

# 5 Simple classifier (logistic regression)
classifier = LogisticRegression(max_iter=100)
train_sizes = [50, 100, 1000, 50000]
accuracy_scores = []

for size in train_sizes:
    # Select a subset of data for training
    subset_indices = np.random.choice(len(X_train_data), size=size, replace=False)
    X_subset = X_train_data[subset_indices]
    y_subset = y_train_data[subset_indices]

    # Train the classifier
    classifier.fit(X_subset, y_subset)

    # Predict on the validation set
    X_val_data = load_images(X_val)
    y_val_pred = classifier.predict(X_val_data)

    # Evaluate accuracy
    accuracy = accuracy_score(y_val, y_val_pred)
    accuracy_scores.append(accuracy)

# Plot the dependence of accuracy on the size of the training set
plt.plot(train_sizes, accuracy_scores, marker='o')
plt.title('Accuracy vs. Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.show()

