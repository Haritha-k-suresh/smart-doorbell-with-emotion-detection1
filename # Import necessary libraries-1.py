# Import necessary libraries
import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Set dataset path
data_path = r"C:\Users\maniv\OneDrive\Desktop\New folder (2)\dataset"
data_dir_list = os.listdir(data_path)

# Image parameters
img_rows, img_cols = 48, 48
num_channel = 3  # RGB images
num_classes = 7  # Number of emotion categories

# Define label dictionary (Ensure proper capitalization)
label_dict = {
    "Angry": 0, "Disgust": 1, "Happy": 2, "Neutral": 3,
    "Sad": 4, "Surprise": 5
}

# Load and preprocess images
img_data_list = []
labels_list = []

for dataset in data_dir_list:
    dataset_title = dataset.capitalize()  # Convert to title case
    if dataset_title not in label_dict:
        print(f"Warning: '{dataset}' not found in label_dict. Skipping!")
        continue  # Skip unknown folders

    img_list = os.listdir(os.path.join(data_path, dataset))
    print(f"Loaded {len(img_list)} images from dataset: {dataset}")

    for img_name in img_list:
        img_path = os.path.join(data_path, dataset, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable image: {img_name}")
            continue  # Skip unreadable images

        img_resized = cv2.resize(img, (img_rows, img_cols))  # Resize to 48x48
        img_data_list.append(img_resized)
        labels_list.append(label_dict[dataset_title])  # Assign correct label

# Convert to numpy array and normalize
img_data = np.array(img_data_list, dtype="float32") / 255.0
labels = np.array(labels_list)

if img_data.shape[0] == 0:
    raise ValueError("No valid images found in dataset. Please check dataset structure!")

print(f"Final Dataset shape: {img_data.shape}")

# Convert labels to one-hot encoding
Y = to_categorical(labels, num_classes)

# Shuffle and split the dataset
x, y = shuffle(img_data, Y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)

# Build CNN Model
input_layer = Input(shape=(48, 48, 3))

# Conv Block 1
f = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
f = Conv2D(32, (3, 3), activation="relu", padding="same")(f)
f = Dropout(0.3)(f)
f = MaxPooling2D((2, 2))(f)

# Conv Block 2
f = Conv2D(64, (3, 3), activation="relu", padding="same")(f)
f = Conv2D(64, (3, 3), activation="relu", padding="same")(f)
f = Dropout(0.3)(f)
f = MaxPooling2D((2, 2))(f)
f = BatchNormalization()(f)

# Conv Block 3
f = Conv2D(128, (3, 3), activation="relu", padding="same")(f)
f = Conv2D(128, (3, 3), activation="relu", padding="same")(f)
f = Dropout(0.3)(f)
f = MaxPooling2D((2, 2))(f)
f = BatchNormalization()(f)

# Conv Block 4
f = Conv2D(256, (3, 3), activation="relu", padding="same")(f)
f = Dropout(0.3)(f)
f = MaxPooling2D((2, 2))(f)
f = BatchNormalization()(f)

# Fully Connected Layers
f = Flatten()(f)
f = Dropout(0.3)(f)
f = Dense(512, activation="relu")(f)
f = Dropout(0.3)(f)
output_layer = Dense(7, activation="softmax")(f)

# Compile Model
model = Model(inputs=[input_layer], outputs=[output_layer])
model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

# Define callbacks
callbacks = [
    ModelCheckpoint("model.h5", monitor="val_accuracy", save_best_only=True, verbose=1),  # Save as .h5
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=20, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_accuracy", restore_best_weights=True, patience=100),
]

# Train model
epochs = 100
history = model.fit(X_train, y_train, batch_size=7, epochs=epochs, validation_data=(X_test, y_test), callbacks=callbacks)

# Plot training results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss", color="red")
plt.plot(history.history["val_loss"], label="Validation Loss", color="blue")
plt.legend()
plt.title("Train Loss vs Validation Loss")

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy", color="red")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="blue")
plt.legend()
plt.title("Train Accuracy vs Validation Accuracy")

plt.show()

# Evaluate model
score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {score[0]}")
print(f"Test Accuracy: {score[1]}")

# Save final model
model.save("lat_model2.h5")

# Test Prediction
test_image = X_test[0:1]
result = np.argmax(model.predict(test_image), axis=1)
print(f"Predicted Label: {list(label_dict.keys())[list(label_dict.values()).index(result[0])]}")

# Display test images with predictions
predictions = np.argmax(model.predict(X_test[:9]), axis=1)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i])
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Prediction: {list(label_dict.keys())[list(label_dict.values()).index(predictions[i])]}")

plt.show()