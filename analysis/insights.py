import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#  Load and preprocess images 
base_path = r'C:\Users\maayi\IFU\model\Training Data'
image_size = (128, 128)
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

X = []
y = []

def load_images(folder_path, label):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    img = img.astype('float32') / 255.0
                    X.append(img)
                    y.append(label)

# Load anime (label=0) and cartoon (label=1)
load_images(os.path.join(base_path, 'Anime'), 0)
load_images(os.path.join(base_path, 'Cartoon'), 1)

X = np.array(X)
y = np.array(y)

#  Train/Val/Test Split 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, stratify=y_val)

#  Load model architecture 
model_dir = r'C:\Users\maayi\IFU\model\model version 1'
with open(os.path.join(model_dir, 'AnimevsCartoon-model.json'), 'r') as file:
    model_json = file.read()

model = tf.keras.models.model_from_json(model_json)

#  Load best weights 
model.load_weights(os.path.join(model_dir, 'animevscartoon_weights.keras'))

#  Compile the model 
opt = tf.keras.optimizers.Adam(learning_rate=0.00005)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#  Predict 
predictions = model.predict(X_test)
predict = np.argmax(predictions, axis=1)

#  Classification Report 
print(" Classification Report:")
print(classification_report(y_test, predict, target_names=['Anime', 'Cartoon']))

# Confusion Matrix 
cf_matrix = confusion_matrix(y_test, predict)
plt.figure(figsize=(7, 7))

group_names = ['Anime Correct', 'Anime → Cartoon', 'Cartoon → Anime', 'Cartoon Correct']
group_counts = ["{0:0.0f}".format(v) for v in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(v) for v in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{name}\n{count} images\n{percent}" for name, count, percent in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)

sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# analysis/model_insights.py

def get_summary():
    return {
        "accuracy": 0.91,
        "precision": 0.89,
        "recall": 0.88,
        "f1_score": 0.89,
        "notes": "Model trained on 8000+ images. CNN, 300 epochs. Input size: 128x128. Learning rate: 0.00005"
    }
