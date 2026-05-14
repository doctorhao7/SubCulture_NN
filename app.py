"""
Web interface for subcultural identity assessment and mental health prediction.
Users answer questions about their engagement with different subcultures,
and the model predicts various mental health indicators.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import json
import os
import torch
from dataloader.preprocess import YScaler
from config.tasks import TASKS
from config.path import RESULTS
from model.pytorch_model import load_model

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Global model and scaler
MODEL = None
Y_SCALER = None

# Load model on module import (for production deployment)
print("Loading trained PyTorch model...")
model_loaded = load_model_pytorch()

if model_loaded:
    print("✓ Model loaded successfully!")
else:
    print("✗ Model not found. Train the model first with: python train_pytorch.py")
    print("  The app will still run but predictions won't be available.")

# Assessment questions configuration
ASSESSMENT_QUESTIONS = {
    'acg_id': {
        'label': 'Animation/Comics/Games (ACG)',
        'description': 'Engagement with anime, manga, video games, and gaming culture',
        'questions': [
            'I regularly watch anime or read manga',
            'I am part of anime/gaming online communities',
            'I attend anime or gaming conventions',
            'I identify strongly with anime/gaming characters or culture',
            'I spend significant time on gaming or anime-related activities',
        ]
    },
    'idl_id': {
        'label': 'Idol Culture',
        'description': 'Fan engagement with music idols (K-pop, J-pop, etc.)',
        'questions': [
            'I regularly follow idol groups (K-pop, J-pop, etc.)',
            'I attend idol concerts or fan events',
            'I am part of idol fan communities or fan clubs',
            'I collect idol merchandise or memorabilia',
            'I spend significant time on idol-related content and discussion',
        ]
    },
    'sp_id': {
        'label': 'Sports',
        'description': 'Engagement with sports fandoms and communities',
        'questions': [
            'I am a passionate sports fan',
            'I attend sporting events or support local teams',
            'I am part of sports fan communities',
            'I regularly discuss or follow sports news',
            'I identify strongly with my favorite sports team',
        ]
    },
    'fas_id': {
        'label': 'Fashion',
        'description': 'Engagement with fashion subcultures and style communities',
        'questions': [
            'I am interested in fashion trends and styles',
            'I am part of fashion-conscious communities',
            'I express my identity through fashion choices',
            'I follow fashion influencers or brands closely',
            'Fashion is an important part of my self-expression',
        ]
    },
    'hip_id': {
        'label': 'Hip-Hop Culture',
        'description': 'Engagement with hip-hop, rap, and related urban cultures',
        'questions': [
            'I enjoy hip-hop and rap music',
            'I am part of hip-hop culture communities',
            'I engage with hip-hop fashion or lifestyle',
            'I identify with hip-hop cultural values and aesthetics',
            'Hip-hop music or culture is important to me',
        ]
    }
}

# Mental health outcome labels for interpretation
MENTAL_HEALTH_LABELS = {
    'bdi_sum': {
        'name': 'Depression',
        'scale': 'Beck Depression Inventory',
        'range': [0, 63],
        'interpretation': 'Higher scores indicate more severe depressive symptoms'
    },
    'suicid': {
        'name': 'Suicidal Ideation',
        'scale': 'Suicidal Ideation Scale',
        'range': [0, 12],
        'interpretation': 'Higher scores indicate greater suicidal thoughts or intent'
    },
    'anger_state': {
        'name': 'State Anger',
        'scale': 'State-Trait Anger Expression Inventory',
        'range': [10, 40],
        'interpretation': 'Current feelings of anger in response to situations'
    },
    'anger_trait': {
        'name': 'Trait Anger',
        'scale': 'State-Trait Anger Expression Inventory',
        'range': [10, 40],
        'interpretation': 'Dispositional tendency to experience anger'
    },
    'anger_in': {
        'name': 'Anger Suppression',
        'scale': 'Anger Expression',
        'range': [8, 32],
        'interpretation': 'Tendency to internalize and suppress angry feelings'
    },
    'anger_out': {
        'name': 'Anger Expression',
        'scale': 'Anger Expression',
        'range': [9, 36],
        'interpretation': 'Tendency to express anger outwardly'
    },
    'anger_control': {
        'name': 'Anger Control',
        'scale': 'Anger Management',
        'range': [7, 28],
        'interpretation': 'Ability to manage and control angry responses'
    },
    'emp_ec': {
        'name': 'Emotional Contagion',
        'scale': 'Toronto Empathy Questionnaire',
        'range': [7, 35],
        'interpretation': 'Tendency to be affected by others\' emotions'
    },
    'emp_pt': {
        'name': 'Perspective Taking',
        'scale': 'Toronto Empathy Questionnaire',
        'range': [7, 35],
        'interpretation': 'Ability to understand others\' viewpoints'
    },
    'emp_fs': {
        'name': 'Feeling Understood',
        'scale': 'Toronto Empathy Questionnaire',
        'range': [7, 35],
        'interpretation': 'Sense of being understood and validated by others'
    },
    'emp_pd': {
        'name': 'Perceived Response',
        'scale': 'Toronto Empathy Questionnaire',
        'range': [7, 35],
        'interpretation': 'Perception of others\' emotional responses to you'
    },
    'ax_state': {
        'name': 'State Anxiety',
        'scale': 'State-Trait Anxiety Inventory',
        'range': [20, 80],
        'interpretation': 'Anxiety experienced in response to specific situations'
    },
    'ax_trait': {
        'name': 'Trait Anxiety',
        'scale': 'State-Trait Anxiety Inventory',
        'range': [20, 80],
        'interpretation': 'General tendency toward anxiety and worry'
    }
}

def load_model_pytorch():
    """Load trained PyTorch model if available."""
    global MODEL, Y_SCALER

    model_path = os.path.join(RESULTS, 'tmp_pytorch', 'pytorch-100epochs', 'model.pt')
    print(f"Looking for model at: {model_path}")
    print(f"Model file exists: {os.path.exists(model_path)}")

    if not os.path.exists(model_path):
        print("Model file not found!")
        return False

    try:
        print("Loading PyTorch model...")
        # Load PyTorch model
        MODEL = load_model(model_path, input_dims=5, output_dims=13)
        MODEL.eval()
        print("Model loaded successfully!")

        # Initialize Y scaler
        dummy_data = pd.DataFrame()
        for key in TASKS['default']['y'].keys():
            dummy_data[key] = [0]
        Y_SCALER = YScaler(dummy_data, 'default')
        print("Y scaler initialized!")

        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def landing():
    """Render the landing/presentation page."""
    return render_template('presentation.html')

@app.route('/assessment')
def index():
    """Render the main assessment page."""
    return render_template('index.html', questions=ASSESSMENT_QUESTIONS)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict mental health outcomes from subcultural identity scores."""
    try:
        data = request.get_json()

        # Extract identity scores from form (1-7 Likert scale)
        input_names = ['acg_id', 'sp_id', 'idl_id', 'fas_id', 'hip_id']
        input_values = [
            (data.get(name, 4) - 1) / 6  # Normalize 1-7 to 0-1
            for name in input_names
        ]

        if MODEL is None:
            return jsonify({
                'success': False,
                'message': 'Model not loaded. Please train the model first.',
                'predictions': None
            }), 400

        # Make prediction with PyTorch
        input_tensor = torch.FloatTensor([input_values])
        with torch.no_grad():
            output_tensor = MODEL(input_tensor)
        output_normalized = output_tensor.numpy()[0]

        # Rescale outputs to original ranges
        output_ranges = TASKS['default']['y']
        predictions = {}

        for i, key in enumerate(output_ranges.keys()):
            min_val, max_val = output_ranges[key]
            original_value = (output_normalized[i] * (max_val - min_val)) + min_val
            predictions[key] = float(original_value)

        # Prepare detailed response
        results = {}
        for key, value in predictions.items():
            if key in MENTAL_HEALTH_LABELS:
                label_info = MENTAL_HEALTH_LABELS[key]
                min_range, max_range = label_info['range']
                percentage = ((value - min_range) / (max_range - min_range)) * 100
                percentage = max(0, min(100, percentage))  # Clamp to 0-100

                results[key] = {
                    'name': label_info['name'],
                    'value': round(value, 2),
                    'percentage': round(percentage, 1),
                    'range': label_info['range'],
                    'interpretation': label_info['interpretation'],
                    'scale': label_info['scale']
                }

        # Calculate risk levels
        risk_factors = {}
        depression_score = predictions.get('bdi_sum', 0)
        suicidal_score = predictions.get('suicid', 0)
        anxiety_score = predictions.get('ax_trait', 0)

        if depression_score > 20:
            risk_factors['depression'] = 'moderate to severe'
        elif depression_score > 10:
            risk_factors['depression'] = 'mild to moderate'
        else:
            risk_factors['depression'] = 'minimal'

        if suicidal_score > 6:
            risk_factors['suicidal_ideation'] = 'concerning'
        elif suicidal_score > 3:
            risk_factors['suicidal_ideation'] = 'mild'
        else:
            risk_factors['suicidal_ideation'] = 'minimal'

        if anxiety_score > 45:
            risk_factors['anxiety'] = 'high'
        elif anxiety_score > 35:
            risk_factors['anxiety'] = 'moderate'
        else:
            risk_factors['anxiety'] = 'low'

        return jsonify({
            'success': True,
            'predictions': results,
            'risk_factors': risk_factors,
            'input_scores': {
                'acg_id': data.get('acg_id', 4),
                'sp_id': data.get('sp_id', 4),
                'idl_id': data.get('idl_id', 4),
                'fas_id': data.get('fas_id', 4),
                'hip_id': data.get('hip_id', 4),
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Prediction error: {str(e)}',
            'predictions': None
        }), 500

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Check if model is loaded and ready."""
    return jsonify({
        'model_loaded': MODEL is not None,
        'model_ready': MODEL is not None
    })

if __name__ == '__main__':
    print("\nStarting web server...")
    print("Visit: http://localhost:5000")
    app.run(debug=True, port=5000)
