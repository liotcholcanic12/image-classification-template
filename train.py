import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

# ==================================
# DATASET PATH
# ==================================

DATASET_PATH = input(
    "\nDataset folder path (press Enter for './dataset'): "
).strip()

if DATASET_PATH == "":
    DATASET_PATH = "dataset"

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10

MODEL_SAVE_PATH = "saved_model/model.h5"

# ==================================
# VERIFY DATASET
# ==================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_PATH}"
    )

print("\nDataset found:")
print(os.path.abspath(DATASET_PATH))

valid_classes = []

for folder in os.listdir(DATASET_PATH):

    folder_path = os.path.join(
        DATASET_PATH,
        folder
    )

    if not os.path.isdir(folder_path):
        continue

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            )
        )
    ]

    if len(images) > 0:
        valid_classes.append(folder)

if len(valid_classes) < 2:
    raise ValueError(
        "\nDataset is empty or invalid.\n"
        "Provide at least TWO folders containing images."
    )

print("\nClasses found:")
for cls in valid_classes:
    print(f"- {cls}")

# ==================================
# CLEAN OLD SPLIT DATA
# ==================================

if os.path.exists("split_data"):
    shutil.rmtree("split_data")

base_dir = "split_data"

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# ==================================
# TRAIN / TEST SPLIT
# ==================================

for class_name in valid_classes:

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    train_class_dir = os.path.join(
        train_dir,
        class_name
    )

    test_class_dir = os.path.join(
        test_dir,
        class_name
    )

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            )
        )
    ]

    train_images, test_images = train_test_split(
        images,
        test_size=0.2,
        random_state=42
    )

    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_class_dir, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(test_class_dir, img)
        )

print("\nDataset split complete.")

# ==================================
# DATA GENERATORS
# ==================================

train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1 / 255
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ==================================
# MODEL
# ==================================

base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(
        train_generator.num_classes,
        activation="softmax"
    )
])

model.compile(
    optimizer=optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==================================
# TRAIN
# ==================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# ==================================
# SAVE MODEL
# ==================================

os.makedirs(
    "saved_model",
    exist_ok=True
)

model.save(MODEL_SAVE_PATH)

print(f"\nModel saved: {MODEL_SAVE_PATH}")

# ==================================
# EVALUATE
# ==================================

loss, accuracy = model.evaluate(
    test_generator
)

print(
    f"\nTest Accuracy: {accuracy*100:.2f}%"
)

predictions = model.predict(
    test_generator
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = test_generator.classes

class_labels = list(
    test_generator.class_indices.keys()
)

conf_matrix = confusion_matrix(
    true_classes,
    predicted_classes
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=class_labels
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_labels
    )
)

precision = precision_score(
    true_classes,
    predicted_classes,
    average="weighted"
)

recall = recall_score(
    true_classes,
    predicted_classes,
    average="weighted"
)

f1 = f1_score(
    true_classes,
    predicted_classes,
    average="weighted"
)

print("\nResults Summary")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
