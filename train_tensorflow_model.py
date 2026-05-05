# ==============================
# train_tensorflow_model.py
# 功能：
# 使用 TensorFlow / Keras 訓練 5 類 ASL 手勢辨識 CNN 模型
#
# 資料來源：
# data/asl_alphabet_train_mvp/
#
# 預期資料夾結構：
# data/asl_alphabet_train_mvp/
# ├── A/
# ├── B/
# ├── C/
# ├── L/
# └── Y/
#
# 輸出：
# 1. outputs/models/cnn_mvp_model.keras
# 2. outputs/models/cnn_mvp_label_mapping.pkl
# 3. outputs/models/cnn_mvp_label_mapping.json
# 4. outputs/figures/cnn_mvp_training_accuracy.png
# 5. outputs/figures/cnn_mvp_training_loss.png
# 6. outputs/figures/cnn_mvp_confusion_matrix.png
# 7. outputs/figures/cnn_mvp_per_class_accuracy.png
# 8. outputs/figures/cnn_mvp_prediction_confidence_distribution.png
# 9. outputs/figures/cnn_mvp_sample_images.png
# 10. outputs/figures/cnn_mvp_classification_report.txt
# ==============================


from pathlib import Path
import json
import random

import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import tensorflow as tf
from tensorflow.keras import layers, models


# ==============================
# 0. 基本設定
# ==============================

# 固定 random seed，讓每次訓練結果比較穩定
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 目前 MVP 只訓練這 5 類
TARGET_CLASSES = ["A", "B", "C", "L", "Y"]

# 圖片大小
IMG_SIZE = 96

# 訓練參數
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2

# 路徑設定
DATA_DIR = Path("data/asl_alphabet_train_mvp")
MODEL_DIR = Path("outputs/models")
FIG_DIR = Path("outputs/figures")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "cnn_mvp_model.keras"
MAPPING_PKL_PATH = MODEL_DIR / "cnn_mvp_label_mapping.pkl"
MAPPING_JSON_PATH = MODEL_DIR / "cnn_mvp_label_mapping.json"


# ==============================
# 1. 檢查資料夾
# ==============================

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"找不到資料夾：{DATA_DIR}\n"
        "請確認你有 data/asl_alphabet_train_mvp/ 資料夾。"
    )

print("========== Dataset Check ==========")
print("Dataset path:", DATA_DIR)

for cls in TARGET_CLASSES:
    class_dir = DATA_DIR / cls

    if not class_dir.exists():
        raise FileNotFoundError(
            f"找不到類別資料夾：{class_dir}\n"
            f"請確認 data/asl_alphabet_train_mvp/ 裡面有 {cls}/"
        )

    image_count = len(list(class_dir.glob("*")))
    print(f"Class {cls}: {image_count} images")

print("===================================")


# ==============================
# 2. 建立 TensorFlow Dataset
# ==============================

# training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    class_names=TARGET_CLASSES,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

# validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    class_names=TARGET_CLASSES,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\n========== Class Mapping ==========")
for index, name in enumerate(class_names):
    print(f"{index} -> {name}")

print("Number of classes:", num_classes)
print("Image size:", IMG_SIZE)


# ==============================
# 3. 加速資料讀取
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ==============================
# 4. 顯示資料集範例圖片
# ==============================

plt.figure(figsize=(10, 6))

for images, labels in train_ds.take(1):
    for i in range(min(10, images.shape[0])):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")

plt.suptitle("Sample Training Images")
plt.tight_layout()
plt.savefig(FIG_DIR / "cnn_mvp_sample_images.png", dpi=300)
plt.show()


# ==============================
# 5. Data Augmentation
# ==============================

# 這一層會在訓練時隨機做圖片增強
# 目的是讓模型比較不會只記住訓練圖片
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.1),
    ],
    name="data_augmentation",
)


# ==============================
# 6. 建立 CNN 模型
# ==============================

model = models.Sequential(
    [
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # 資料增強
        data_augmentation,

        # 將 pixel 從 0~255 縮放到 0~1
        layers.Rescaling(1.0 / 255),

        # 第一組 Conv Block
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # 第二組 Conv Block
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # 第三組 Conv Block
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # 第四組 Conv Block
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # 把影像特徵攤平成一維
        layers.Flatten(),

        # 全連接層
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        # 輸出層：5 類
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n========== Model Summary ==========")
model.summary()


# ==============================
# 7. 訓練模型
# ==============================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    ),
]

print("\n========== Start Training ==========")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

print("========== Training Finished ==========")


# ==============================
# 8. 評估模型
# ==============================

val_loss, val_acc = model.evaluate(val_ds)

print("\n========== Validation Result ==========")
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_acc)


# ==============================
# 9. 收集 validation 預測結果
# ==============================

y_true = []
y_pred = []
y_confidence = []

for images, labels in val_ds:
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds)
    y_confidence.extend(confs)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_confidence = np.array(y_confidence)


# ==============================
# 10. Classification Report
# ==============================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
)

print("\n========== Classification Report ==========")
print(report)

with open(FIG_DIR / "cnn_mvp_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)


# ==============================
# 11. 畫 Accuracy 圖
# ==============================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN MVP Training Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIG_DIR / "cnn_mvp_training_accuracy.png", dpi=300)
plt.show()


# ==============================
# 12. 畫 Loss 圖
# ==============================

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN MVP Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIG_DIR / "cnn_mvp_training_loss.png", dpi=300)
plt.show()


# ==============================
# 13. Confusion Matrix
# ==============================

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("CNN MVP Confusion Matrix")
plt.tight_layout()
plt.savefig(FIG_DIR / "cnn_mvp_confusion_matrix.png", dpi=300)
plt.show()


# ==============================
# 14. Per-class Accuracy
# ==============================

per_class_accuracy = []

for i, cls in enumerate(class_names):
    class_indices = np.where(y_true == i)[0]

    if len(class_indices) == 0:
        acc = 0
    else:
        acc = np.mean(y_pred[class_indices] == y_true[class_indices])

    per_class_accuracy.append(acc)

plt.figure(figsize=(8, 5))
plt.bar(class_names, per_class_accuracy)
plt.ylim(0, 1)
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.title("CNN MVP Per-Class Accuracy")
plt.tight_layout()
plt.savefig(FIG_DIR / "cnn_mvp_per_class_accuracy.png", dpi=300)
plt.show()

print("\n========== Per-Class Accuracy ==========")
for cls, acc in zip(class_names, per_class_accuracy):
    print(f"{cls}: {acc:.4f}")


# ==============================
# 15. Confidence Distribution
# ==============================

plt.figure(figsize=(8, 5))
plt.hist(y_confidence, bins=20)
plt.xlabel("Prediction Confidence")
plt.ylabel("Count")
plt.title("CNN MVP Prediction Confidence Distribution")
plt.tight_layout()
plt.savefig(FIG_DIR / "cnn_mvp_prediction_confidence_distribution.png", dpi=300)
plt.show()


# ==============================
# 16. 儲存模型與 Label Mapping
# ==============================

model.save(MODEL_PATH)

mapping_data = {
    "class_names": class_names,
    "img_size": IMG_SIZE,
    "target_classes": TARGET_CLASSES,
}

joblib.dump(mapping_data, MAPPING_PKL_PATH)

with open(MAPPING_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(mapping_data, f, indent=4, ensure_ascii=False)

print("\n========== Saved Files ==========")
print("Model:", MODEL_PATH)
print("Label mapping pkl:", MAPPING_PKL_PATH)
print("Label mapping json:", MAPPING_JSON_PATH)
print("Figures saved to:", FIG_DIR)
print("=================================")