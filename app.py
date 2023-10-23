import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Function to load the FER2013 dataset
def load_fer2013_dataset(dataset_dir):
    X = []  # List to store image data
    y = []  # List to store corresponding labels

    # Define emotion labels based on FER2013 dataset
    emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    for emotion_label in emotion_labels:
        # Construct the path to the folder containing images of the current emotion
        emotion_dir = os.path.join(dataset_dir, emotion_label)

        # Loop through images in the folder
        for image_filename in os.listdir(emotion_dir):
            # Load and preprocess the image
            image_path = os.path.join(emotion_dir, image_filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
            image = cv2.resize(image, (48, 48))  # Resize to a common size (adjust as needed)

            # Append the image data and corresponding label
            X.append(image)
            y.append(emotion_labels.index(emotion_label))

    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y

# Define the path to your FER2013 dataset folder
dataset_dir = "C:/Users/91950/Downloads/app (1)/train"

# Load the dataset
X, y = load_fer2013_dataset(dataset_dir)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize pixel values (typically scaled between 0 and 1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define your CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')  # 7 emotions
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',  # Since y is not one-hot encoded
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use the default camera (you can specify a different camera if needed)

# Define emotion labels for displaying predictions
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Read the music recommendations from the CSV file
music_recommendations_csv = "C:/Users/91950/Downloads/app (1)/mu.xlsx"  # Adjust the filename and path as needed
music_recommendations_df = pd.read_excel(music_recommendations_csv)

# Initialize a logistic regression model
logistic_regression_model = LogisticRegression()
'''def apply_cartoon_filter(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    gray = cv2.medianBlur(gray, 5)

    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # Apply a bilateral filter to smooth the image while preserving edges
    color = cv2.bilateralFilter(frame, 9, 300, 300)

    # Combine the color image and edges using bitwise and
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    return cartoon'''
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    '''if not ret:
        print("Error: Could not read frame")
        continue'''
    #cartoon_frame = apply_cartoon_filter(frame)
    # Preprocess the frame (resize and convert to grayscale)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.resize(frame_gray, (48, 48))

    # Normalize pixel values (same as training data)
    frame_gray = frame_gray / 255.0

    # Make a prediction using the CNN model
    emotion_id = np.argmax(model.predict(np.expand_dims(frame_gray, axis=0)))
    predicted_emotion = emotion_labels[emotion_id]

    # Perform binary classification using logistic regression
    # Define a binary outcome based on your criteria (e.g., "Positive" or "Negative")
    # Modify this logic based on your criteria
    binary_outcome = 0  # Default to "Negative" (e.g., ["Fear", "Angry", "Sad"])
    if predicted_emotion in ["Happy", "Surprise", "Disgust", "Neutral"]:
        binary_outcome = 1  # Set to "Positive"

    # Display both the emotion and binary outcome on the frame
    cv2.putText(frame, f"Emotion: {predicted_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Binary Outcome: {binary_outcome}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Find music recommendations based on the detected emotion
    recommendations_for_emotion = music_recommendations_df[predicted_emotion].dropna().tolist()

    print(f"Top 3 Music Recommendations for {predicted_emotion}:")
    for i, recommendation in enumerate(recommendations_for_emotion[:3]):
        print(f"{i+1}. {recommendation}")

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
