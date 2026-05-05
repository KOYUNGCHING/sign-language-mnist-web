# ==============================
# 功能：
# 使用 ASL Alphabet 圖片資料集訓練 Scikit-learn baseline model
#
# 模型：
# Random Forest Classifier
#
# 輸入資料：
# data/asl_alphabet_train/
#
# 輸出：
# outputs/models/sklearn_random_forest.pkl
# outputs/models/sklearn_label_mapping.pkl
# outputs/figures/sklearn_*.png
# ==============================

from pathlib import Path
import random
import json

import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/asl_alphabet_train")
TEST_DIR = Path("data/asl_alphabet_test")

FIG_DIR = Path("outputs/figures")
MODEL_DIR = Path("outputs/models")

FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Scikit-learn 版不要一次吃完整 87,000 張，會比較慢
# MVP 建議 200~500
# 如果想用全部資料，改成 None
MAX_IMAGES_PER_CLASS = 300

# Scikit-learn baseline 使用灰階 + 64x64
IMAGE_SIZE = 64

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# 1. 檢查資料夾
if not DATA_DIR.exists():
    raise FileNotFoundError(f"找不到訓練資料夾：{DATA_DIR}")

class_names = sorted([
    folder.name
    for folder in DATA_DIR.iterdir()
    if folder.is_dir()
])

if len(class_names) == 0:
    raise ValueError(f"{DATA_DIR} 裡沒有任何類別資料夾")

print("========== Class Names ==========")
print(class_names)
print("Number of classes:", len(class_names))

class_to_index = {name: idx for idx, name in enumerate(class_names)}
index_to_class = {idx: name for name, idx in class_to_index.items()}


# 2. 載入圖片並轉成特徵
def load_image_as_feature(image_path: Path) -> np.ndarray:
    """
    將圖片轉成 Scikit-learn 可用的特徵向量。

    步驟：
    1. 讀取圖片
    2. 轉灰階
    3. resize 成 64x64
    4. normalize 到 0~1
    5. flatten 成 4096 維
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"無法讀取圖片：{image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    normalized = resized.astype("float32") / 255.0

    return normalized.flatten()


X = []
y = []
image_paths_record = []

print("\n========== Loading Training Images ==========")

for class_name in class_names:
    class_folder = DATA_DIR / class_name

    image_paths = list(class_folder.glob("*.jpg")) + \
                  list(class_folder.glob("*.jpeg")) + \
                  list(class_folder.glob("*.png"))

    image_paths = sorted(image_paths)

    if MAX_IMAGES_PER_CLASS is not None:
        image_paths = image_paths[:MAX_IMAGES_PER_CLASS]

    print(f"{class_name}: {len(image_paths)} images")

    for image_path in image_paths:
        try:
            feature = load_image_as_feature(image_path)
            X.append(feature)
            y.append(class_to_index[class_name])
            image_paths_record.append(str(image_path))
        except Exception as e:
            print(f"Skip {image_path}: {e}")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("\n========== Dataset Shape ==========")
print("X shape:", X.shape)
print("y shape:", y.shape)


# 3. Label distribution 圖
counts = np.bincount(y, minlength=len(class_names))

plt.figure(figsize=(14, 5))
plt.bar(class_names, counts)
plt.xticks(rotation=45)
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Scikit-learn Training Data Distribution")
plt.tight_layout()
plt.savefig(FIG_DIR / "sklearn_label_distribution.png", dpi=300)
plt.show()


# 4. Sample images 圖
plt.figure(figsize=(12, 8))

sample_per_class = []

for class_name in class_names:
    folder = DATA_DIR / class_name
    imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
    if imgs:
        sample_per_class.append((class_name, imgs[0]))

num_show = min(24, len(sample_per_class))

for i in range(num_show):
    class_name, image_path = sample_per_class[i]
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(4, 6, i + 1)
    plt.imshow(image_rgb)
    plt.title(class_name)
    plt.axis("off")

plt.suptitle("Sample Images from ASL Alphabet Dataset", fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "sklearn_sample_images.png", dpi=300)
plt.show()


# 5. 切分 train / validation
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y,
)

print("\n========== Train / Validation Split ==========")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)


# 6. 訓練 Random Forest
print("\n========== Training Random Forest ==========")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=1,
)

rf_model.fit(X_train, y_train)


# 7. 評估 validation set
print("\n========== Evaluating Random Forest ==========")

y_pred = rf_model.predict(X_val)
val_acc = accuracy_score(y_val, y_pred)

print("Validation Accuracy:", val_acc)

report = classification_report(
    y_val,
    y_pred,
    target_names=class_names,
)

print("\nClassification Report:")
print(report)

with open(FIG_DIR / "sklearn_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)


# 8. Confusion Matrix
cm = confusion_matrix(y_val, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

fig, ax = plt.subplots(figsize=(14, 14))
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues")
plt.title("Scikit-learn Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig(FIG_DIR / "sklearn_confusion_matrix.png", dpi=300)
plt.show()


# 9. Per-class accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(14, 5))
plt.bar(class_names, per_class_acc)
plt.ylim(0, 1.05)
plt.xticks(rotation=45)
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.title("Scikit-learn Random Forest Per-Class Accuracy")

for i, acc in enumerate(per_class_acc):
    plt.text(i, acc + 0.02, f"{acc:.2f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "sklearn_per_class_accuracy.png", dpi=300)
plt.show()


# 10. 儲存模型與 mapping
joblib.dump(rf_model, MODEL_DIR / "sklearn_random_forest.pkl")

mapping_data = {
    "class_names": class_names,
    "class_to_index": class_to_index,
    "index_to_class": index_to_class,
    "image_size": IMAGE_SIZE,
}

joblib.dump(mapping_data, MODEL_DIR / "sklearn_label_mapping.pkl")

with open(MODEL_DIR / "sklearn_label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(mapping_data, f, indent=2)

print("\n========== Saved Files ==========")
print("Model:", MODEL_DIR / "sklearn_random_forest.pkl")
print("Mapping:", MODEL_DIR / "sklearn_label_mapping.pkl")
print("Figures:", FIG_DIR)