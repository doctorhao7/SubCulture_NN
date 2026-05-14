# Does Anime, Idol Culture Bring Depression? Structural Analysis and Deep Learning on Subcultural Identity and Various Psychological Outcomes

This repository implements deep learning models to predict mental health outcomes based on subcultural identity measures. It addresses the research question: does engagement with anime, idol culture, and other subcultures correlate with depression and other psychological outcomes?

## Research Background

Recent studies show that identity related to anime, idol, and hip-hop cultures is positively associated with anxiety, aggression, depressive symptoms, and suicidal tendencies in youth. However, the mechanisms are complex—social support and perceived reputation within these communities play mediating roles. This project uses neural networks to predict psychological outcomes from subcultural identity measures without requiring clinical diagnosis, helping to destigmatize these communities.

## Data Overview

### Input Variables (Subcultural Identity Measures)
The model takes 5 subcultural identity dimensions as input, each scaled to 0-1:

- **acg_id** (Animation/Comics/Games identity): Range [1, 7]
  - Measures engagement and identification with anime, manga, and gaming subcultures
- **sp_id** (Sports identity): Range [1, 7]
  - Measures sports fandom and athletic community engagement
- **idl_id** (Idol culture identity): Range [1, 7]
  - Measures fan devotion to music idols (primarily K-pop and J-pop)
- **fas_id** (Fashion identity): Range [1, 7]
  - Measures fashion subculture engagement
- **hip_id** (Hip-hop identity): Range [1, 7]
  - Measures hip-hop and rap culture engagement

**Note:** All inputs are normalized using Min-Max scaling: `(value - min) / (max - min)`

### Output Variables (Mental Health Measures)
The model predicts 13 psychological outcome measures:

**Depression & Suicidality:**
- **bdi_sum**: Beck Depression Inventory total score, Range [0, 63]
  - Comprehensive assessment of depressive symptoms
- **suicid**: Suicidal ideation score, Range [0, 12]
  - Measures thoughts and intentions toward self-harm

**Anger Measures (State-Trait Anger Expression Inventory):**
- **anger_state**: State anger, Range [10, 40]
  - Immediate, situational anger
- **anger_trait**: Trait anger, Range [10, 40]
  - Dispositional tendency toward anger
- **anger_in**: Anger suppression (inward), Range [8, 32]
  - Tendency to internalize angry feelings
- **anger_out**: Anger expression (outward), Range [9, 36]
  - Tendency to express anger openly
- **anger_control**: Anger control/management, Range [7, 28]
  - Ability to regulate anger responses

**Empathy Measures (Toronto Empathy Questionnaire):**
- **emp_ec**: Emotional contagion, Range [7, 35]
  - Tendency to be affected by others' emotions
- **emp_pt**: Perspective taking, Range [7, 35]
  - Ability to understand others' viewpoints
- **emp_fs**: Feeling of being understood, Range [7, 35]
  - Sense of being comprehended by others
- **emp_pd**: Perceived emotional response, Range [7, 35]
  - Perception of others' emotional reactions

**Anxiety:**
- **ax_state**: State anxiety, Range [20, 80]
  - Temporary anxiety in response to situations
- **ax_trait**: Trait anxiety, Range [20, 80]
  - Dispositional anxiety proneness

**Note:** All outputs are normalized using Min-Max scaling to [0, 1] range.

## Model Architecture

### Current Model Selection: Dense Neural Network (32-32 architecture)

**Why this architecture?**
- The relationship between subcultural identity and mental health is complex but relatively continuous
- A two-layer dense network with 32 units per layer provides sufficient capacity without overfitting on moderate-sized datasets
- Dropout regularization (0.2) prevents overfitting and improves generalization

**Architecture Details:**
```
Input Layer: 5 features (normalized subcultural identity scores)
    ↓
Hidden Layer 1: 32 units, ReLU activation
    ↓
Dropout: 0.2 regularization
    ↓
Hidden Layer 2: 32 units, ReLU activation
    ↓
Dropout: 0.2 regularization
    ↓
Output Layer: 13 units, Linear activation (no activation for regression)
    ↓
Output: 13 mental health measure scores (normalized to [0, 1])
```

**Hyperparameters:**
- **Loss Function**: Mean Squared Error (MSE)
  - Chosen for regression task predicting continuous psychological scores
- **Optimizer**: Adam
  - Adaptive learning rate provides robust convergence
- **Metrics**: MSE and Accuracy
  - MSE measures prediction error magnitude
  - Accuracy tracks the proportion of correct predictions
- **Batch Size**: 40
  - Balances gradient noise and computational efficiency
- **Epochs**: 1000 (default, adjustable per model)
  - Early stopping via model checkpointing saves best model by loss

### Model Selection Rationale

The project includes infrastructure to test multiple architectures:

| Architecture | Layers | Dropout | Best For |
|---|---|---|---|
| Single hidden layer [16] | 1 × 16 units | 0.2 | Small datasets, simple relationships |
| Single hidden layer [32] | 1 × 32 units | 0.2 | Baseline with moderate capacity |
| Single hidden layer [64] | 1 × 64 units | 0.2 | Datasets with higher complexity |
| **Two hidden layers [32, 32]** | **2 × 32 units** | **0.2** | **Current choice: good capacity/regularization balance** |
| Three hidden layers [32, 32, 32] | 3 × 32 units | 0.2 | Complex nonlinear relationships (risk of overfitting) |

## Training & Evaluation

### Data Split
- **Training Set**: 80% of data (random shuffle)
- **Validation Set**: 20% of data (used for performance monitoring)

### Training Procedure
1. Models are trained for specified epochs (default: 1000)
2. TensorBoard logs histogram statistics every epoch for analysis
3. ModelCheckpoint callback saves weights of best-performing model (by loss)
4. Early stopping via checkpointing prevents overfitting

### Files Generated
- `results/tmp/{model_name}-{epochs}epochs/logs/` — TensorBoard event files
- `results/tmp/{model_name}-{epochs}epochs/SubNet` — Best trained weights

## Usage

### Training a Model
```python
python train.py
```

Configuration in `train.py`:
- `EPOCHS`: Number of training epochs (default: 1000)
- `INPUT_DIMS`: Number of input features (5 subcultural identities)
- `OUTPUT_DIMS`: Number of output variables (13 mental health measures)
- `TASK`: Task configuration name ('default' uses TASKS['default'] from config/tasks.py)
- `OUTPUT_FOLDER`: Subfolder for results (default: 'tmp')

### Making Predictions
```python
python predict.py
```

Load trained model and predict mental health outcomes from subcultural identity scores.

## Configuration

### `config/tasks.py`
Defines datasets and prediction tasks:
- **x**: Input variable ranges (subcultural identities)
- **y**: Output variable ranges (mental health measures)
- **models**: List of neural network architectures to train

### `config/path.py`
Update data and results directories:
```python
ROOT = r'D:\Saves\SubCulture'  # Change to your machine's path
```

## Dataset Format

Required CSV columns:
- Subcultural identity: `acg_id`, `sp_id`, `idl_id`, `fas_id`, `hip_id`
- Mental health outcomes: `bdi_sum`, `suicid`, `anger_state`, `anger_trait`, `anger_in`, `anger_out`, `anger_control`, `emp_ec`, `emp_pt`, `emp_fs`, `emp_pd`, `ax_state`, `ax_trait`
- Additional columns (demographic, other measures) are ignored

Example:
```csv
ID,age,gender,...,acg_id,sp_id,idl_id,fas_id,hip_id,...,bdi_sum,suicid,anger_state,...
```

## Interpretation

The model outputs normalized predictions [0, 1] which must be rescaled to the original scale:
```
actual_value = (normalized_value * (max - min)) + min
```

Higher values indicate:
- **bdi_sum, suicid, anger_*, ax_***: Worse psychological outcomes
- **emp_***: Higher empathy levels

## References

Liu, Y., et al. (2022). Does anime, idol culture bring depression? Structural analysis and deep learning on subcultural identity and various psychological outcomes. *Heliyon*, 8(9), e10607.
https://doi.org/10.1016/j.heliyon.2022.e10607
