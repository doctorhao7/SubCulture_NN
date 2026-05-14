"""
Batch prediction script: predict mental health outcomes from subcultural identity scores.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from config.path import RESULTS, get_data_file
from config.tasks import TASKS
import os


def load_trained_model(model_path):
    """Load trained model weights."""
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_shape=(5,), dtype='float64'))
    model.add(Dropout(rate=0.2, dtype='float64'))
    model.add(Dense(units=32, activation='relu', dtype='float64'))
    model.add(Dropout(rate=0.2, dtype='float64'))
    model.add(Dense(units=13, activation=None, dtype='float64'))

    model.load_weights(model_path)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    return model


def predict_from_scores(acg_id, sp_id, idl_id, fas_id, hip_id, model_path=None):
    """
    Predict mental health outcomes from subcultural identity scores (1-7 scale).
    """
    # Default model path
    if model_path is None:
        model_path = os.path.join(RESULTS, 'tmp', '32-32-100epochs', 'SubNet.weights.h5')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first with: python train.py")

    # Load model
    model = load_trained_model(model_path)

    # Normalize inputs from 1-7 to 0-1
    input_values = np.array([
        (acg_id - 1) / 6,
        (sp_id - 1) / 6,
        (idl_id - 1) / 6,
        (fas_id - 1) / 6,
        (hip_id - 1) / 6
    ], dtype=np.float64)

    # Make prediction
    input_array = np.array([input_values], dtype=np.float64)
    output_normalized = model.predict(input_array, verbose=0)[0]

    # Rescale outputs to original ranges
    output_ranges = TASKS['default']['y']
    predictions = {}

    for i, key in enumerate(output_ranges.keys()):
        min_val, max_val = output_ranges[key]
        original_value = (output_normalized[i] * (max_val - min_val)) + min_val
        predictions[key] = float(original_value)

    return predictions


if __name__ == '__main__':
    print("Subcultural Identity & Mental Health Prediction")
    print("=" * 50)

    # Example: Predict for single person
    print("\nExample: Predict for individual with specific identity scores")
    print("-" * 50)

    predictions = predict_from_scores(
        acg_id=6,      # High ACG identity
        sp_id=3,       # Low sports identity
        idl_id=5,      # Moderate-high idol identity
        fas_id=4,      # Moderate fashion identity
        hip_id=2       # Low hip-hop identity
    )

    print("\nIdentity Scores: ACG=6, Sports=3, Idol=5, Fashion=4, Hip-hop=2")
    print("\nPredicted Mental Health Outcomes:")
    print("-" * 50)

    output_ranges = TASKS['default']['y']

    for key, value in predictions.items():
        min_range, max_range = output_ranges[key]
        percentage = ((value - min_range) / (max_range - min_range)) * 100
        percentage = max(0, min(100, percentage))
        print(f"{key:20} = {value:6.2f} (Range: {min_range:3}-{max_range:3}, {percentage:5.1f}%)")
