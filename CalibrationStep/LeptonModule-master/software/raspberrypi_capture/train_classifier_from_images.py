import cv2                             # OpenCV for image processing
import numpy as np                     # NumPy for numerical operations
import os                              # For file and directory operations
from sklearn.svm import SVC            # Support Vector Classifier
import joblib                          # For saving and loading models
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning
from sklearn.preprocessing import StandardScaler  # For feature normalization

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
    # Get original dimensions
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)             # Compute the center of the image

    # 1) Compute the rotation matrix around the center
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 2) Calculate the sine and cosine of rotation to find new bounds
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    # Compute new width and height to ensure the entire image fits
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 3) Adjust the translation component of the matrix
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 4) Perform the affine warp with reflection border to avoid black edges
    return cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

# === Augmentation Function ===
def augment_image(img):
    """
    Generate augmented versions of the input image,
    including flips, fixed rotations, and arbitrary rotations.
    """
    # Base augmentations: original, flips, 90° and 180° rotations
    augmented = [
        img,
        cv2.flip(img, 1),                                # Horizontal flip
        cv2.flip(img, 0),                                # Vertical flip
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),        # Rotate 90° clockwise
        cv2.rotate(img, cv2.ROTATE_180),                 # Rotate 180°
    ]

    # Arbitrary-angle rotations (±15°, ±30°, ±45°, ±60°, ±75°)
    angles = [15, 30, 45, 60, 75]
    for a in angles:
        augmented.append(rotate_image(img,  a))          # Positive angle
        augmented.append(rotate_image(img, -a))          # Negative angle

    # Additional augmentations: blur, brightness adjustments, histogram equalization
    augmented += [
        cv2.GaussianBlur(img, (5, 5), 0),                # Gaussian blur
        cv2.add(img, 15),                                # Increase brightness
        cv2.subtract(img, 15),                           # Decrease brightness
        cv2.equalizeHist(img) if len(img.shape) == 2 else img  # Histogram equalization (grayscale only)
    ]
    return augmented

# === Feature Extraction ===
def extract_features_from_image_array(img):
    """
    Extract geometric and invariant features from a binary thresholded image.

    Returns:
        feature_vector (ndarray) or None if no valid contour found.
    """
    # 1) Binarize using Otsu's threshold
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 2) Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None  # No contours detected

    # 3) Keep the largest contour by area
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:
        return None  # Skip small noise contours

    # 4) Geometric features from bounding box
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    aspect_ratio = float(w) / h
    extent = area / (w * h)

    # 5) Convex hull solidity
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area != 0 else 0

    # 6) Hu Moments (log-transformed for scale invariance)
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    # 7) Combine all features into a single vector
    return np.concatenate([hu_log, [aspect_ratio, extent, solidity]])

# === Prepare Training Data ===
data_dir = "/home/pi/Documents/CalibrationStep/LeptonModule-master/software/raspberrypi_capture/data/train"
X = []                                  # Feature matrix
y = []                                  # Labels

# Map folder names (class labels) to integer indices
labels = sorted(os.listdir(data_dir))
label_map = {label: i for i, label in enumerate(labels)}

# Iterate over each class folder and image file
for label in labels:
    folder = os.path.join(data_dir, label)
    for img_file in os.listdir(folder):
        path = os.path.join(folder, img_file)
        # Read as grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Could not read image: {path}")
            continue

        # Resize to a consistent target size
        TARGET_SIZE = (320, 240)
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        # Augment and extract features for each variant
        for aug_img in augment_image(img):
            features = extract_features_from_image_array(aug_img)
            if features is not None:
                X.append(features)
                y.append(label_map[label])

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# === Normalize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)      # Zero-mean, unit-variance scaling

# === Train Classifier with Hyperparameter Tuning ===
param_grid = {
    'C': [0.1, 1, 10, 100],             # Regularization strength
    'gamma': ['scale', 0.01, 0.001],    # Kernel coefficient
    'kernel': ['rbf']                   # Radial Basis Function kernel
}

# 5-fold cross-validated grid search optimizing accuracy
grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_scaled, y)

# Best estimator after grid search
clf = grid.best_estimator_
print("✅ Best parameters:", grid.best_params_)
print("✅ Trained on full dataset (augmented + normalized)")

# === Save Model, Scaler & Label Map ===
joblib.dump(clf, "svm_hu_classifier.pkl")    # Save the trained SVM model
joblib.dump(label_map, "label_map.pkl")      # Save the label-to-index mapping
joblib.dump(scaler, "scaler.pkl")            # Save the feature scaler
print("✅ Classifier, label map, and scaler saved.")
