# EigenFaces with OpenCV – Simplified Example

This example demonstrates face recognition using Eigenfaces computed via OpenCV's PCA functions. The implementation is based on our main code.

## Step 1: Import Libraries

```python
import cv2  # OpenCV library
import numpy as np
import os    # for file path handling
import matplotlib.pyplot as plt  # For visualization
```

## Step 2: Load and Preprocess Images

We use the function `load_image_from_image_dataset` defined in our main code. This function reads images from subfolders (one per person) and resizes them.

```python
def load_image_from_image_dataset(folder_path, target_size=(64, 64)):
    images = []
    labels = []
    label_map = {}
    label_id = 0

    for folder_name in os.listdir(folder_path):
        subject_path = os.path.join(folder_path, folder_name)
        if not os.path.isdir(subject_path):
            continue

        if folder_name not in label_map:
            label_map[folder_name] = label_id
            label_id += 1

        for filename in os.listdir(subject_path):
            img_path = os.path.join(subject_path, filename)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read image {filename}")
                    continue
                img_resized = cv2.resize(img, target_size)
                images.append(img_resized.flatten())
                labels.append(label_map[folder_name])
            except Exception as e:
                print(f"Error loading image {filename}: {e}")

    return np.array(images).T, labels, label_map

train_folder = "face_dataset"  # Folder with subfolders per person
train_images, labels, label_map = load_image_from_image_dataset(train_folder)
```

## Step 3: Training – Compute Eigenfaces

Using OpenCV's PCA, we compute the mean face and eigenfaces. Note that before applying PCA, we reshape the image matrix.

```python
def train_eigenfaces(image_matrix):
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(image_matrix.T, None)
    return mean, eigenvectors, eigenvalues

reshaped_train_images = train_images.reshape(train_images.shape[0] * train_images.shape[1],
                                              train_images.shape[2])
mean_face, eigenface_opencv, eigenvalues = train_eigenfaces(reshaped_train_images)
```

## Step 4: Project Training Faces

We project each training image into the eigenface space using the `project_face` function.

```python
def project_face(face_image, mean_face, eigenfaces):
    centered_face = face_image - mean_face.flatten()
    projection = np.dot(centered_face, eigenfaces.T)
    return projection

projected_reshaped_train_face = []
for face_image in reshaped_train_images.T:
    projection = project_face(face_image, mean_face, eigenface_opencv)
    projected_reshaped_train_face.append(projection)
projected_reshaped_train_face = np.array(projected_reshaped_train_face)
```

## Step 5: Face Recognition

For a given test image, we project it and compare the projection with the training images. We use a simple Euclidean distance metric. Note that the label lookup uses a reverse mapping of label_map.

```python
def recognize_face(test_image_path, mean_face, eigenfaces, projected_train_faces, train_labels, target_size=(64, 64)):
    try:
        test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        if test_img is None:
            print(f"Error: Could not read test image {test_image_path}")
            return "Unknown"

        test_img_resized = cv2.resize(test_img, target_size)
        test_face_vector = test_img_resized.flatten()

        projected_test_face = project_face(test_face_vector, mean_face, eigenfaces)

        min_distance = float('inf')
        predicted_label = "Unknown"

        for i, projected_face in enumerate(projected_train_faces):
            distance = np.linalg.norm(projected_test_face - projected_face)
            if distance < min_distance:
                min_distance = distance
                predicted_label = train_labels[i]

        return predicted_label, min_distance

    except Exception as e:
        print(f"Error during recognition: {e}")
        return "Error", float('inf')

test_image_path = "/Users/kumarswamikallimath/Desktop/EigenFaces/test_image.jpg"
labels_rev_num = {v: k for k, v in label_map.items()}
predicted_label, distance = recognize_face(test_image_path, 
                                           mean_face, 
                                           eigenface_opencv,
                                           projected_reshaped_train_face,
                                           labels_rev_num)
print(f"Predicted label for {test_image_path}: {predicted_label}, Distance: {distance}")
```

## Optional: Visualize Mean Face and Eigenfaces

```python
plt.imshow(mean_face.reshape(64, 64), cmap="gray")
plt.title("Mean Face")
plt.axis("off")
plt.show()

plt.figure(figsize=(25,25))
for i in range(eigenface_opencv.shape[0]):
    plt.subplot(5,3,i+1)
    eigenface = eigenface_opencv[i,:].reshape(64,64)
    plt.imshow(eigenface, cmap="gray")
    plt.title(f"Eigenface {i+1}")
    plt.axis("off")
plt.suptitle("Top Eigenfaces (OpenCV PCA)")
plt.tight_layout()
plt.show()
```

The functions `load_image_from_image_dataset`, `train_eigenfaces`, `project_face`, and `recognize_face` are defined in the main code. Adjust folder paths as needed.