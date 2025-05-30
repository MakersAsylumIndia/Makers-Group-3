# This is a supervised machine learning script.
# It uses a classification algorithm (K-Nearest Neighbors, KNN) to learn from labeled sensor data.
# The model is trained to classify sign language gestures based on sensor readings.

import json
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib # For saving/loading the model
import os

# --- Configuration ---
MAPPINGS_FILE = "sign_language_mappings_cli.json" # From your CLI app
MODEL_FILE = "sign_language_knn_model.joblib"
SCALER_FILE = "sign_language_scaler.joblib"
LABEL_ENCODER_FILE = "sign_language_label_encoder.joblib"
NUM_SENSORS = 5 # Must match your data collection

# --- Data Loading and Preprocessing ---
def load_data(filepath=MAPPINGS_FILE):
    """Loads gesture data from the JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: Data file '{filepath}' not found.")
        return None, None
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filepath}'.")
        return None, None
    except Exception as e:
        print(f"Error loading data file '{filepath}': {e}")
        return None, None

    features = []
    labels = []
    for letter, readings_list in data.items():
        if isinstance(readings_list, list) and len(readings_list) == NUM_SENSORS:
            features.append(readings_list)
            labels.append(letter)
        else:
            print(f"Warning: Skipping invalid data for letter '{letter}': {readings_list}")

    if not features or not labels:
        print("No valid data loaded to process.")
        return None, None

    return np.array(features), np.array(labels)

def preprocess_data(features, labels):
    """Prepares data for model training: scaling features and encoding labels."""
    if features is None or labels is None:
        return None, None, None, None

    # Scale features (important for distance-based algorithms like KNN)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Encode labels (convert letters like 'A', 'B' to numbers 0, 1, ...)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    return features_scaled, labels_encoded, scaler, label_encoder

# --- Model Training ---
def train_model(features_scaled, labels_encoded, n_neighbors=3):
    """Trains a KNN model."""
    if features_scaled is None or labels_encoded is None:
        print("Error: Cannot train model with no data.")
        return None

    if len(features_scaled) < n_neighbors:
        print(f"Warning: Number of samples ({len(features_scaled)}) is less than n_neighbors ({n_neighbors}).")
        print("Consider reducing n_neighbors or collecting more diverse data for each class.")
        if len(features_scaled) < 1:
             print("Cannot train with 0 samples.")
             return None
        n_neighbors = max(1, len(features_scaled) -1) # Adjust n_neighbors if possible
        if n_neighbors == 0 and len(features_scaled) == 1: # Handle case of only 1 sample
            n_neighbors = 1
        print(f"Adjusted n_neighbors to {n_neighbors}")


    # Simple train-test split to evaluate
    # For very small datasets, cross-validation is better, or train on all data
    if len(np.unique(labels_encoded)) > 1 and len(labels_encoded) > n_neighbors * 2 : # Ensure enough samples and classes for split
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
        )
    else: # Train on all data if dataset is too small or only one class
        print("Dataset too small for train/test split or only one class. Training on all available data.")
        X_train, y_train = features_scaled, labels_encoded
        X_test, y_test = None, None # No test set in this case


    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel trained. Test Accuracy: {accuracy:.4f}")
        print("Classification Report (on test set):")
        # Need to convert numeric labels back to original letter labels for the report
        try:
            # Get label_encoder from the preprocessing step or pass it
            # For now, assuming we have access or it's re-fitted if necessary
            # This part is tricky if label_encoder isn't directly available here.
            # For simplicity, let's assume it's passed or we show numeric.
            # A better way is to pass the label_encoder to train_model.
            # For now, we'll just show numeric labels if we can't decode.
            # target_names_str = label_encoder.inverse_transform(np.unique(np.concatenate((y_test, y_pred))))
            # print(classification_report(y_test, y_pred, target_names=target_names_str, zero_division=0))
             print(classification_report(y_test, y_pred, zero_division=0)) # Simpler, shows numeric labels
        except Exception as e:
            print(f"Could not generate full classification report: {e}")
            print(f"Numeric y_test: {np.unique(y_test)}")
            print(f"Numeric y_pred: {np.unique(y_pred)}")

    else:
        print("\nModel trained on all available data (no test set evaluation).")

    # Cross-validation (optional, good for small datasets)
    if len(np.unique(labels_encoded)) > 1 and len(labels_encoded) > n_neighbors: # Ensure enough samples/classes
        try:
            scores = cross_val_score(model, features_scaled, labels_encoded, cv=min(3, len(np.unique(labels_encoded)), len(features_scaled)//n_neighbors if len(features_scaled)//n_neighbors > 0 else 1), scoring='accuracy')
            print(f"Cross-validation Accuracy Scores: {scores}")
            print(f"Average CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        except ValueError as ve:
            print(f"Could not perform cross-validation: {ve}. Likely too few samples or classes.")
    else:
        print("Skipping cross-validation due to insufficient data or single class.")


    return model

# --- Model Persistence ---
def save_model_pipeline(model, scaler, label_encoder,
                        model_path=MODEL_FILE, scaler_path=SCALER_FILE, label_encoder_path=LABEL_ENCODER_FILE):
    """Saves the trained model, scaler, and label encoder."""
    if model:
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    if scaler:
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
    if label_encoder:
        joblib.dump(label_encoder, label_encoder_path)
        print(f"Label encoder saved to {label_encoder_path}")

def load_model_pipeline(model_path=MODEL_FILE, scaler_path=SCALER_FILE, label_encoder_path=LABEL_ENCODER_FILE):
    """Loads a trained model, scaler, and label encoder."""
    model, scaler, label_encoder = None, None, None
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model file {model_path} not found.")

        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        else:
            print(f"Scaler file {scaler_path} not found.")

        if os.path.exists(label_encoder_path):
            label_encoder = joblib.load(label_encoder_path)
            print(f"Label encoder loaded from {label_encoder_path}")
        else:
            print(f"Label encoder file {label_encoder_path} not found.")

    except Exception as e:
        print(f"Error loading model/pipeline components: {e}")
        return None, None, None # Return None for all if any part fails

    if not model or not scaler or not label_encoder:
        print("One or more essential components (model, scaler, label_encoder) could not be loaded.")
        return None, None, None

    return model, scaler, label_encoder

# --- Prediction ---
def predict_gesture(sensor_readings, model, scaler, label_encoder):
    """Predicts the gesture letter for new sensor readings."""
    if model is None or scaler is None or label_encoder is None:
        print("Error: Model, scaler, or label encoder not loaded. Cannot predict.")
        return None, None

    if not isinstance(sensor_readings, list) or len(sensor_readings) != NUM_SENSORS:
        print(f"Error: Invalid sensor readings format. Expected list of {NUM_SENSORS} numbers.")
        return None, None

    try:
        # Reshape for single sample and scale
        readings_np = np.array(sensor_readings).reshape(1, -1)
        readings_scaled = scaler.transform(readings_np)

        # Predict
        prediction_encoded = model.predict(readings_scaled)
        prediction_proba = model.predict_proba(readings_scaled) # Get probabilities

        # Decode prediction
        predicted_letter = label_encoder.inverse_transform(prediction_encoded)[0]
        confidence = np.max(prediction_proba)

        return predicted_letter, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# --- Main Execution for Training (Example) ---
if __name__ == "__main__":
    print("--- Sign Language Gesture ML Training ---")

    # 1. Load Data
    features, labels = load_data()

    if features is not None and labels is not None:
        print(f"Loaded {len(features)} samples with {len(np.unique(labels))} unique labels: {np.unique(labels)}")

        # 2. Preprocess Data
        features_scaled, labels_encoded, scaler, label_encoder = preprocess_data(features, labels)

        if features_scaled is not None:
            # 3. Train Model
            # You can experiment with n_neighbors
            # For small datasets, a smaller k (like 1 or 3) might be better.
            # Ensure you have at least k samples per class ideally, or enough total samples.
            k = min(3, len(features_scaled)) if len(features_scaled) > 0 else 1
            if len(np.unique(labels)) == 1: # If only one class, k must be 1
                k = 1
            elif k == 0: # If features_scaled is empty for some reason.
                print("No features to train on.")
                exit()


            print(f"Using n_neighbors = {k} for KNN.")
            model = train_model(features_scaled, labels_encoded, n_neighbors=k)

            if model:
                # 4. Save Model, Scaler, and Label Encoder
                save_model_pipeline(model, scaler, label_encoder)

                print("\n--- Example Prediction (using loaded model) ---")
                # 5. Load the pipeline (model, scaler, label_encoder)
                loaded_model, loaded_scaler, loaded_label_encoder = load_model_pipeline()

                if loaded_model and loaded_scaler and loaded_label_encoder:
                    # Create some dummy sensor data for testing prediction
                    # (should be similar to what your ESP32 would send)
                    if len(features) > 0:
                        example_sensor_data = list(features[0]) # Use first sample from loaded data
                        actual_letter = labels[0]
                        print(f"\nPredicting for example data (similar to '{actual_letter}'): {example_sensor_data}")
                        predicted_sign, confidence = predict_gesture(example_sensor_data, loaded_model, loaded_scaler, loaded_label_encoder)

                        if predicted_sign:
                            print(f"Predicted Sign: {predicted_sign} (Confidence: {confidence:.2f})")
                            print(f"Actual Sign was: {actual_letter}")
                        else:
                            print("Prediction failed for the example.")

                        # Example of slightly different data
                        example_noisy_data = [x + np.random.randint(-15, 15) for x in example_sensor_data]
                        print(f"\nPredicting for noisy data: {example_noisy_data}")
                        predicted_sign_noisy, confidence_noisy = predict_gesture(example_noisy_data, loaded_model, loaded_scaler, loaded_label_encoder)
                        if predicted_sign_noisy:
                             print(f"Predicted Sign (noisy): {predicted_sign_noisy} (Confidence: {confidence_noisy:.2f})")
                        else:
                            print("Prediction failed for noisy example.")

                    else:
                        print("No data available to create an example prediction.")
                else:
                    print("Could not load model pipeline for example prediction.")
            else:
                print("Model training failed.")
        else:
            print("Data preprocessing failed.")
    else:
        print("Data loading failed. Ensure 'sign_language_mappings_cli.json' exists and has data.")