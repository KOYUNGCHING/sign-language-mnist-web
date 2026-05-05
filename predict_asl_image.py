# ==============================
# 功能：
# 使用 5 類 CNN MVP 模型預測單張 ASL 圖片
#
# 輸入：
# test_asl.jpg
#
# 輸出：
# outputs/figures/cnn_mvp_single_prediction.png
# ==============================

from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

#  路徑設定
MODEL_PATH = Path("outputs/models/cnn_mvp_model.keras")
MAPPING_PATH = Path("outputs/models/cnn_mvp_label_mapping.pkl")

# 想測哪個字母就改這裡
TEST_LETTER = "C"

IMAGE_PATH = Path(f"data/asl_alphabet_test/{TEST_LETTER}_test.jpg")

FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# 1. 檢查檔案
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "找不到 outputs/models/cnn_mvp_model.keras，請先執行 train_tensorflow_cnn_mvp.py"
    )

if not MAPPING_PATH.exists():
    raise FileNotFoundError(
        "找不到 outputs/models/cnn_mvp_label_mapping.pkl，請先執行 train_tensorflow_cnn_mvp.py"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        "找不到 test_asl.jpg，請放一張測試圖片到專案根目錄，或修改 IMAGE_PATH。"
    )


# 2. 載入模型與 mapping
model = tf.keras.models.load_model(MODEL_PATH)
mapping_data = joblib.load(MAPPING_PATH)

class_names = mapping_data["class_names"]
IMG_SIZE = mapping_data["img_size"]

print("Model loaded:", MODEL_PATH)
print("Classes:", class_names)
print("Image size:", IMG_SIZE)


# 3. 圖片前處理
def preprocess_image(image_path: Path):
    """
    將單張圖片轉成 CNN 模型輸入格式。

    步驟：
    1. OpenCV 讀取圖片
    2. BGR 轉 RGB
    3. Resize 成訓練時的 IMG_SIZE
    4. 加上 batch 維度
    """

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise ValueError(f"OpenCV 無法讀取圖片：{image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))

    input_tensor = resized_rgb.astype("float32")
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return image_rgb, resized_rgb, input_tensor


image_rgb, resized_rgb, input_tensor = preprocess_image(IMAGE_PATH)


# 4. 預測
probs = model.predict(input_tensor)
pred_index = int(np.argmax(probs[0]))
confidence = float(np.max(probs[0]))
pred_class = class_names[pred_index]

print("\n========== Prediction Result ==========")
print("Predicted class:", pred_class)
print("Confidence:", confidence)


# 5. Top 5 Predictions
top_k = min(5, len(class_names))
top_indices = np.argsort(probs[0])[::-1][:top_k]

top_classes = [class_names[int(i)] for i in top_indices]
top_scores = [float(probs[0][i]) for i in top_indices]

print("\nTop Predictions:")
for rank, (cls, score) in enumerate(zip(top_classes, top_scores), start=1):
    print(f"{rank}. {cls}: {score:.4f}")


# 6. 畫一些圖
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(resized_rgb.astype("uint8"))
plt.title(f"Resized {IMG_SIZE}x{IMG_SIZE}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.bar(top_classes, top_scores)
plt.ylim(0, 1)
plt.xlabel("Class")
plt.ylabel("Probability")
plt.title("Top Predictions")

plt.suptitle(f"Prediction: {pred_class} ({confidence:.2f})", fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "cnn_mvp_single_prediction.png", dpi=300)
plt.show()