import cv2
import numpy as np
import os
from sklearn.svm import SVC
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# === Utility: Arbitrary-angle rotation ===
def rotate_image(img, angle):
    """
    Rotate an image around its center without clipping the corners.

    Args:
        img   (ndarray): input image
        angle (float):   rotation angle in degrees. Positive = counter-clockwise.

    Returns:
        rotated (ndarray): the rotated image, sized to fit the whole rotated frame.
    """
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)

    # 1) get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 2) compute new bounding dimensions
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 3) adjust translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 4) warp
    return cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

# === Augmentation Function ===
def augment_image(img):
    """
    Generate augmented versions of the input image,
    including arbitrary rotations.
    """
    # Base augmentations
    augmented = [
        img,
        cv2.flip(img, 1),                                # Horizontal flip
        cv2.flip(img, 0),                                # Vertical flip
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),        # Rotate 90
        cv2.rotate(img, cv2.ROTATE_180),                 # Rotate 180
    ]

    # Arbitrary-angle rotations
    angles = [15, 30, 45, 60, 75]
    for a in angles:
        augmented.append(rotate_image(img,  a))
        augmented.append(rotate_image(img, -a))

    # Other augmentations
    augmented += [
        cv2.GaussianBlur(img, (5, 5), 0),                # Blurred
        cv2.add(img, 15),                                # Brighter
        cv2.subtract(img, 15),                           # Darker
        cv2.equalizeHist(img) if len(img.shape) == 2 else img  # Histogram equalization
    ]
    return augmented


# === Feature Extraction ===
def extract_features_from_image_array(img):
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:  # Skip tiny junk
        return None

    # Geometric features
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    aspect_ratio = float(w) / h
    extent = area / (w * h)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area != 0 else 0

    # Hu Moments (log scale)
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    # Combine features
    return np.concatenate([hu_log, [aspect_ratio, extent, solidity]])

# === Prepare Training Data ===
data_dir = "/home/pi/Documents/CalibrationStep/LeptonModule-master/software/raspberrypi_capture/data/train"
X = []
y = []

labels = sorted(os.listdir(data_dir))
label_map = {label: i for i, label in enumerate(labels)}

for label in labels:
    folder = os.path.join(data_dir, label)
    for img_file in os.listdir(folder):
        path = os.path.join(folder, img_file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Could not read image: {path}")
            continue

        TARGET_SIZE = (320, 240)
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        for aug_img in augment_image(img):
            features = extract_features_from_image_array(aug_img)
            if features is not None:
                X.append(features)
                y.append(label_map[label])

X = np.array(X)
y = np.array(y)

# === Normalize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train Classifier ===
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_scaled, y)

clf = grid.best_estimator_
print("✅ Best parameters:", grid.best_params_)
print("✅ Trained on full dataset (augmented + normalized)")

# === Save Model, Scaler & Labels ===
joblib.dump(clf, "svm_hu_classifier.pkl")
joblib.dump(label_map, "label_map.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Classifier, label map, and scaler saved.")
