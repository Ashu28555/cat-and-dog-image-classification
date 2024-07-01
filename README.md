# cat-and-dog-image-classification
# Task :  Implement a support vector machine (SVM) to classify images of cats and dogs from the kaggle dataset.

!pip install opendatasets
import opendatasets as od
import pandas

od.download('https://www.kaggle.com/shaunthesheep/microsoft-catsvsdogs-dataset')

pip install numpy opencv-python scikit-image scikit-learn


import opendatasets as od
import zipfile
import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

BASE_DIR ="/kaggle/input/microsoft-catsvsdogs-dataset/PetImages/"

dataset_path = 'microsoft-catsvsdogs-dataset/PetImages'

extracted_path = os.path.join('microsoft-catsvsdogs-dataset', 'PetImages')

cats_path = os.path.join(extracted_path, 'Cat/')
dogs_path = os.path.join(extracted_path, 'Dog/')

def load_images(path, label):
    images = []
    labels = []
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            try:
                img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (128, 128))  # Resize to 64x64
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error reading file {file}: {e}")
    return images, labels

cat_images, cat_labels = load_images(cats_path, 0)
dog_images, dog_labels = load_images(dogs_path, 1)

cat_images

cat_labels

dog_images

dog_labels

np.array(cat_images).shape

np.array(dog_images).shape

images = np.array(cat_images + dog_images)
labels = np.array(cat_labels + dog_labels)

def extract_hog_features(images):
    hog_features = []
    for image in images:
        features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_features.append(features)
    return np.array(hog_features)

hog_features = extract_hog_features(images)

hog_features

x_train, x_val, y_train, y_val = train_test_split(hog_features, labels, test_size=0.2, random_state=42)

svm = LinearSVC(max_iter=10000)
svm.fit(x_train, y_train)

y_pred = svm.predict(x_val)

y_pred

print(f"Accuracy: {accuracy_score(y_val, y_pred)}")

print(classification_report(y_val, y_pred, target_names=['Cat', 'Dog']))

cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cat', 'Dog'])
disp.plot()

import numpy as np
import matplotlib.pyplot as plt

# Choose a random image from the dataset
image_index = np.random.randint(len(images))
image = images[image_index]

# Predict the class of the image
predicted_class = svm.predict([hog_features[image_index]])

# Display the image and the prediction
plt.imshow(image, cmap="gray")
plt.title(f"Prediction: {predicted_class[0]}", fontsize=15)
plt.axis("off")
plt.show()
if predicted_class[0] == 0:
    print("This is a Cat")
else:
    print("This is a Dog")




import pickle

# Save the SVM model
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)


import pickle

# Load the SVM model
with open('svm_model.pkl', 'rb') as f:
    loaded_svm = pickle.load(f)




# Now you can use this loaded model for predictions
predicted_class = loaded_svm.predict([hog_features[image_index]])

# Display the image and the prediction
plt.imshow(image, cmap="gray")
plt.title(f"Prediction: {predicted_class[0]}", fontsize=15)
plt.axis("off")
plt.show()

if predicted_class[0] == 0:
    print("This is a Cat")
else:
    print("This is a Dog")


# The End
