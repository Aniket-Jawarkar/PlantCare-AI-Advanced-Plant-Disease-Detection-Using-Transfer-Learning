# 3_Model_Training/train_model.py
import os
import sys
import json
import tensorflow as tf

# allow imports from 2_Model_Building
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "2_Model_Building"))

import config
from build_model import build_model
from utils import get_num_classes

# ---- Settings (you can tweak) ----
EPOCHS = 10               # adjust if you want fewer/more epochs for CPU
BATCH_SIZE = config.BATCH_SIZE
IMG_SIZE = config.IMG_SIZE
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "6_Models_and_Outputs")
BEST_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "plant_disease_best.h5")
HISTORY_JSON = os.path.join(MODEL_OUTPUT_DIR, "history.json")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "6_Models_and_Outputs", "logs")

# Make sure output dirs exist
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---- Data generators (augmentation for train) ----
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0
)

train_dir = config.TRAIN_DIR
val_dir = config.VALID_DIR

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_data.num_classes
print(f"Detected {num_classes} classes, {train_data.samples} train images, {val_data.samples} validation images")

# ---- Build & compile model ----
model = build_model(num_classes, input_shape=(*IMG_SIZE, 3), base_trainable=False)
model.summary()

# ---- Callbacks ----
callbacks = [
    # Save best model by val_accuracy
    tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    # Reduce LR when val_loss plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-7
    ),

    # Early stopping to avoid long CPU runs
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),

    # Optional: csv logger
    tf.keras.callbacks.CSVLogger(
        os.path.join(MODEL_OUTPUT_DIR, "training_log.csv")
    ),

    # Optional: TensorBoard (useful if you open locally)
    tf.keras.callbacks.TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=0
    )
]

# ---- Train ----
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---- Save final model (best model already saved by checkpoint) ----
final_model_path = os.path.join(MODEL_OUTPUT_DIR, "plant_disease_final.h5")
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")
print(f"Best model saved to: {BEST_MODEL_PATH}")

# ---- Save history to JSON (convert numpy types to python floats) ----
hist = {}
for k, v in history.history.items():
    hist[k] = [float(x) for x in v]

with open(HISTORY_JSON, "w") as f:
    json.dump(hist, f)
print(f"Training history saved to: {HISTORY_JSON}")