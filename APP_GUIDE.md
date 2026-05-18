# Subcultural Identity Assessment Web Application

## Overview

This web application provides an interactive interface for assessing how engagement with different subcultures relates to mental health outcomes. Users answer questions about their involvement with anime/games, idol culture, sports, fashion, and hip-hop, then receive personalized predictions of various mental health indicators.

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow/Keras installed
- Flask installed

### Setup

```bash
# Install dependencies
pip install tensorflow flask pandas numpy

# Navigate to project directory
cd /Users/howliu/Developer/SubCulture_NN
```

## Running the Application

### Step 1: Train the Model

```bash
python train.py
```

This will:
- Load dataset300.csv (expanded to 300 records from the original 100)
- Split into 80% training and 20% validation
- Train multiple neural network architectures
- Save the best model weights to `results/tmp/32-32-100epochs/SubNet.weights.h5`

Expected training time: ~2-5 minutes (for 100 epochs, adjust `EPOCHS` in train.py for more/less training)

### Step 2: Start the Web Server

```bash
python app.py
```

The server will:
- Load the trained model
- Start a Flask development server on http://localhost:5000
- Print confirmation messages

Sample output:
```
Loading trained model...
✓ Model loaded successfully!

Starting web server...
Visit: http://localhost:5000
 * Running on http://127.0.0.1:5000
```

### Step 3: Open the Assessment

Open your web browser and navigate to:
```
http:///xxxxxx
```

## Using the Assessment

### Taking the Assessment

1. **Read the Introduction**
   - Understand the purpose and disclaimer
   - Note that this is for educational/research purposes, not clinical diagnosis

2. **Rate Your Subcultural Engagement** (5 sections)
   - Each section has a 1-7 Likert scale slider
   - 1 = Strongly disagree with statements about that subculture
   - 7 = Strongly agree with statements about that subculture
   - Sections: ACG (Anime/Comics/Games), Idol Culture, Sports, Fashion, Hip-Hop

3. **View Results**
   - Click "Get Assessment Results"
   - Review risk overview (depression, anxiety, suicidal ideation)
   - Explore detailed mental health indicators with interpretations

### Interpreting Results

**Risk Levels:**
- **Minimal/Low**: Scores in the lower range, fewer concerning psychological outcomes
- **Mild/Moderate**: Moderate levels of psychological concerns
- **Moderate to Severe/High**: Higher levels requiring professional attention

**Mental Health Measures:**

| Measure | Scale | Interpretation |
|---------|-------|---|
| **Depression (BDI)** | 0-63 | Higher = more depressive symptoms |
| **Suicidal Ideation** | 0-12 | Higher = more concerning |
| **State Anger** | 10-40 | Current anger level |
| **Trait Anger** | 10-40 | Dispositional anger tendency |
| **Anger Suppression** | 8-32 | Tendency to internalize anger |
| **Anger Expression** | 9-36 | Tendency to express anger outwardly |
| **Anger Control** | 7-28 | Ability to manage anger |
| **Empathy (Emotional Contagion)** | 7-35 | Affected by others' emotions |
| **Empathy (Perspective Taking)** | 7-35 | Understanding others' viewpoints |
| **Empathy (Feeling Understood)** | 7-35 | Feeling validated by others |
| **Empathy (Perceived Response)** | 7-35 | Others' emotional responses to you |
| **State Anxiety** | 20-80 | Current anxiety level |
| **Trait Anxiety** | 20-80 | Dispositional anxiety tendency |

## Making Batch Predictions

To predict outcomes for a CSV file of participants:

```bash
python predict.py
```

This will make predictions for sample identity scores and show the output.

## Architecture Details

### Model Architecture
```
Input (5 features) 
  → Dense(32, ReLU) → Dropout(0.2)
  → Dense(32, ReLU) → Dropout(0.2)
  → Dense(13, Linear)
Output (13 predictions)
```

### Input Features (Normalized 0-1)
- Animation/Comics/Games identity
- Sports identity
- Idol culture identity
- Fashion identity
- Hip-hop identity

### Output Features (13 mental health measures)
- Depression (BDI sum)
- Suicidal ideation
- Anger: state, trait, suppression, expression, control
- Empathy: emotional contagion, perspective, feeling understood, perceived response
- Anxiety: state, trait

## Troubleshooting

### Model Not Found
```
Error: Model not loaded. Please train the model first.
```
**Solution:** Run `python train.py` first

### TensorFlow Import Error
```
ModuleNotFoundError: No module named 'tensorflow'
```
**Solution:** Install TensorFlow: `pip install tensorflow`

### Flask Import Error
```
ModuleNotFoundError: No module named 'flask'
```
**Solution:** Install Flask: `pip install flask`

### Port Already in Use
```
Error: Address already in use
```
**Solution:** 
- Change the port in app.py: `app.run(debug=True, port=5001)`
- Or kill the process using port 5000: `lsof -i :5000` then `kill -9 <PID>`

### Connection Refused
```
Error: Failed to connect to localhost:5000
```
**Solution:** Ensure app.py is running: `python app.py`

## Advanced Configuration

### Training Hyperparameters
Edit `train.py`:
```python
EPOCHS = 1000              # Number of training epochs
INPUT_DIMS = 5             # Number of input features
OUTPUT_DIMS = 13           # Number of output predictions
BATCH_SIZE = 40            # Batch size for training
```

### Model Architectures
Edit `config/tasks.py`:
```python
TASKS['default']['models'] = [
    {'layers': [32, 32], 'dropout': 0.2},  # Current
    {'layers': [64, 64], 'dropout': 0.3},  # Alternative
]
```

### Flask Settings
Edit `app.py`:
```python
app.run(debug=True, port=5000, host='0.0.0.0')  # Allow remote access
```

## Important Disclaimers

🚨 **This tool is NOT a clinical diagnostic instrument**

- Results are statistical predictions, not medical diagnoses
- Should not be used as the sole basis for mental health decisions
- If experiencing mental health concerns, please consult a qualified mental health professional
- Cultural engagement is complex and multifactorial
- Individual circumstances and contexts matter greatly

## Research Citation

This implementation is based on the research:

Liu, Y., et al. (2022). Does anime, idol culture bring depression? Structural analysis and deep learning on subcultural identity and various psychological outcomes. *Heliyon*, 8(9), e10607.
https://doi.org/10.1016/j.heliyon.2022.e10607

## Support & Issues

For technical issues or improvements:
1. Check the troubleshooting section above
2. Review the main README.md for model architecture details
3. Verify all dependencies are correctly installed
