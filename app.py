import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ============================================================
# SETTINGS
# ============================================================

DATASET = "Bone_Fracture_Binary_Classification"

TRAIN_DIR = os.path.join(DATASET, "train")
TEST_DIR = os.path.join(DATASET, "test")

IMG_SIZE = (64, 64)

# Number of images used from each class.
# Increase later if you want.
MAX_TRAIN_IMAGES = 500
MAX_TEST_IMAGES = 200


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(folder, max_images):

    X = []
    y = []

    classes = {
        "not fractured": 0,
        "fractured": 1
    }

    for class_name, label in classes.items():

        class_path = os.path.join(
            folder,
            class_name
        )

        print()
        print("Loading:", class_path)

        if not os.path.exists(class_path):

            print("ERROR: Folder not found!")
            continue

        files = os.listdir(class_path)

        # Limit number of images for faster training
        files = files[:max_images]

        count = 0

        for filename in files:

            filepath = os.path.join(
                class_path,
                filename
            )

            image = cv2.imread(
                filepath,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            # Resize
            image = cv2.resize(
                image,
                IMG_SIZE
            )

            # Normalize pixel values
            image = image.astype(
                np.float32
            ) / 255.0

            # Convert image to one-dimensional data
            features = image.flatten()

            X.append(features)
            y.append(label)

            count += 1

        print(
            "Images loaded:",
            count
        )

    return np.array(X), np.array(y)


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_xray(image):

    # Grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Gaussian blur
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Morphological processing
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    erosion = cv2.erode(
        blurred,
        kernel,
        iterations=1
    )

    dilation = cv2.dilate(
        erosion,
        kernel,
        iterations=1
    )

    # Canny edge detection
    edges = cv2.Canny(
        dilation,
        50,
        150
    )

    # Find contours
    contours, _ = cv2.findContours(
        edges.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Copy original
    result = image.copy()

    # Draw contours
    cv2.drawContours(
        result,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    return result, edges


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    original,
    highlighted,
    edges,
    prediction,
    confidence
):

    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    highlighted_rgb = cv2.cvtColor(
        highlighted,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(
        figsize=(16, 6)
    )

    # Original
    plt.subplot(
        1,
        3,
        1
    )

    plt.title(
        "Original X-ray"
    )

    plt.imshow(
        original_rgb
    )

    plt.axis("off")


    # Edges
    plt.subplot(
        1,
        3,
        2
    )

    plt.title(
        "Canny Edge Detection"
    )

    plt.imshow(
        edges,
        cmap="gray"
    )

    plt.axis("off")


    # Final
    plt.subplot(
        1,
        3,
        3
    )

    plt.title(
        f"{prediction}\nConfidence: {confidence:.2f}%"
    )

    plt.imshow(
        highlighted_rgb
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()


# ============================================================
# START PROGRAM
# ============================================================

print()
print("==========================================")
print("          BONE FRACTURE AI")
print("==========================================")
print()

print("Dataset:", DATASET)

# ============================================================
# LOAD TRAINING DATA
# ============================================================

print()
print("Loading training dataset...")

X_train, y_train = load_dataset(
    TRAIN_DIR,
    MAX_TRAIN_IMAGES
)

print()
print(
    "Total training images:",
    len(X_train)
)


if len(X_train) == 0:

    print()
    print("ERROR: No training images found.")
    print()
    print("Check this folder:")
    print(TRAIN_DIR)

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# LOAD TEST DATA
# ============================================================

print()
print("Loading test dataset...")

X_test, y_test = load_dataset(
    TEST_DIR,
    MAX_TEST_IMAGES
)

print()
print(
    "Total test images:",
    len(X_test)
)


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("==========================================")
print("             TRAINING AI")
print("==========================================")
print()

print("Please wait...")

model = SVC(
    kernel="rbf",
    probability=True
)

model.fit(
    X_train,
    y_train
)

print()
print("Training completed!")


# ============================================================
# TEST MODEL
# ============================================================

if len(X_test) > 0:

    print()
    print("Testing model...")

    test_predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    print()
    print("==========================================")
    print("             MODEL RESULT")
    print("==========================================")

    print(
        "Test Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )

    print("==========================================")


# ============================================================
# SAVE MODEL
# ============================================================

with open(
    "fracture_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )

print()
print("Model saved as:")
print("fracture_model.pkl")


# ============================================================
# ASK FOR USER X-RAY
# ============================================================

print()
print("==========================================")
print("             YOUR X-RAY")
print("==========================================")
print()

image_path = input(
    "Enter your X-ray image path: "
).strip()


# ============================================================
# LOAD USER IMAGE
# ============================================================

image = cv2.imread(
    image_path
)

if image is None:

    print()
    print("ERROR: Could not open image.")
    print()
    print("Example:")
    print("my_xray.jpg")

    input("\nPress Enter to exit...")
    exit()


print()
print("X-ray loaded successfully!")


# ============================================================
# PREPARE IMAGE FOR AI
# ============================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

resized = cv2.resize(
    gray,
    IMG_SIZE
)

normalized = resized.astype(
    np.float32
) / 255.0

features = normalized.flatten()

features = features.reshape(
    1,
    -1
)


# ============================================================
# AI PREDICTION
# ============================================================

prediction = model.predict(
    features
)[0]

probabilities = model.predict_proba(
    features
)[0]

confidence = max(
    probabilities
) * 100


if prediction == 1:

    result = "FRACTURED"

else:

    result = "NOT FRACTURED"


# ============================================================
# DISPLAY TEXT RESULT
# ============================================================

print()
print()
print("==========================================")
print("          BONE FRACTURE AI")
print("==========================================")
print()
print(
    "Prediction:",
    result
)
print()
print(
    "Confidence:",
    round(confidence, 2),
    "%"
)
print()
print("==========================================")


# ============================================================
# IMAGE PROCESSING
# ============================================================

highlighted, edges = process_xray(
    image
)


# ============================================================
# SAVE RESULT
# ============================================================

cv2.imwrite(
    "fracture_result.jpg",
    highlighted
)

print()
print("Processed image saved as:")
print("fracture_result.jpg")


# ============================================================
# SHOW IMAGES
# ============================================================

display_result(
    image,
    highlighted,
    edges,
    result,
    confidence
)