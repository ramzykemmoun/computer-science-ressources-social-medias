import os
import numpy as np
from PIL import Image
from collections import Counter


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def load_image(path, size=(64, 64)):
    """
    Load an image, convert to grayscale, resize,
    normalize, and flatten into a 1D vector.
    """
    image = Image.open(path).convert("L")
    image = image.resize(size)

    image_array = np.array(image, dtype=float)
    image_array = image_array / 255.0  # normalize [0,1]

    return image_array.flatten()


# --------------------------------------------------
# DATASET LOADING
# --------------------------------------------------

def load_dataset(dataset_path):
    """
    Dataset structure:
    dataset/
        class_1/
        class_2/
        class_3/

    Returns:
    X -> image vectors
    y -> labels
    """
    X = []
    y = []

    for label in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, label)

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            img_vector = load_image(img_path)

            X.append(img_vector)
            y.append(label)

    return np.array(X), np.array(y)


# --------------------------------------------------
# DISTANCE FUNCTION (FROM SCRATCH)
# --------------------------------------------------

def euclidean_distance(a, b):
    """
    Euclidean distance:
    sqrt( sum( (a - b)^2 ) )
    """
    return np.sqrt(np.sum((a - b) ** 2))


# --------------------------------------------------
# KNN CLASSIFIER (FROM SCRATCH)
# --------------------------------------------------

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        kNN does not train.
        It only stores the dataset.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            predictions.append(self._predict_one(x))
        return predictions

    def _predict_one(self, x):
        distances = []

        for i in range(len(self.X_train)):
            dist = euclidean_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda item: item[0])
        k_nearest = distances[:self.k]

        labels = [label for _, label in k_nearest]
        return Counter(labels).most_common(1)[0][0]


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():
    DATASET_PATH = "dataset"
    TEST_IMAGE_PATH = "test_image.jpg"

    print("Loading dataset...")
    X, y = load_dataset(DATASET_PATH)
    print(f"Loaded {len(X)} images")

    model = KNN(k=5)
    model.fit(X, y)

    print("Predicting test image...")
    test_vector = load_image(TEST_IMAGE_PATH)
    prediction = model.predict([test_vector])

    print("Prediction:", prediction[0])


if __name__ == "__main__":
    main()