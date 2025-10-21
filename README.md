
## 🎯 Overview

This project aims to automatically predict the **Mean Opinion Score (MOS)** — a subjective measure of perceived web page quality — using machine learning and deep learning techniques. By analyzing user behavioral data (clicks, time spent, scroll patterns, etc.), we can estimate user satisfaction without intrusive surveys.

**Key Objectives:**
- Predict MOS scores (1-5 scale) from user interaction data
- Compare classical ML and modern DL approaches
- Provide insights for UX designers and web developers

## ✨ Features

- **Comprehensive Data Preprocessing**: Handling missing values, feature engineering, and normalization
- **Multiple Model Architectures**: Implementation of 6 different models
- **Advanced Metrics**: MAE, MSE, RMSE, and R² evaluation
- **Visualization Tools**: Learning curves, residual analysis, and prediction scatter plots
- **Feature Engineering**: Custom MOS estimation based on behavioral metrics

## 📊 Dataset

The dataset contains **10,376 records** with the following features:

| Feature | Description |
|---------|-------------|
| Website | Website identifier |
| Category | Website category (194 unique categories) |
| Rank Change | Position change in rankings |
| Avg. Visit Duration | Average time spent on site |
| Pages / Visit | Average pages viewed per session |
| Bounce Rate | Percentage of single-page visits |
| Fetched From | Data source |
| Top-Level Category | Main category (23 categories) |

**Data Source**: [Dataset Link](https://www.similarweb.com/top-websites/) (April 2024)

### Statistical Summary

```
Avg. Visit Duration: 4.31 min (σ=5.02)
Pages / Visit: 4.54 (σ=4.10)
Bounce Rate: 0.45 (σ=0.15)
Rank Change: 5.90 (σ=23.95)
```

## 🤖 Models

### Machine Learning
1. **Random Forest**: Ensemble method with decision trees

### Deep Learning
2. **Autoencoder**: Dimensionality reduction and feature extraction
3. **MLP (Multi-Layer Perceptron)**: Standard feedforward neural network
4. **Autoencoder + Random Forest**: Hybrid approach combining feature extraction and ensemble learning
5. **MLP + Autoencoder**: Deep neural network with learned representations
6. **TabNet**: Attention-based architecture for tabular data
7. **Wide & Deep Learning**: Combined memorization and generalization

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda




### Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
torch>=1.9.0
pytorch-tabnet>=3.1.0
jupyter>=1.0.0
```

## 💻 Usage

### Data Preprocessing

```python
from src.preprocessing import load_and_preprocess_data

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/web_traffic.csv')
```

### Training Models

```python
from src.models import MLPAutoencoder

# Initialize and train model
model = MLPAutoencoder(input_dim=3, hidden_dims=[64, 32, 16])
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions
predictions = model.predict(X_test)
```

### Running Experiments

```bash
# Train all models
python train.py --config configs/default.yaml

# Evaluate models
python evaluate.py --model mlp_autoencoder --checkpoint checkpoints/best_model.pth

# Generate visualizations
python visualize.py --results results/metrics.json
```

### Jupyter Notebooks

Explore the interactive notebooks in the `notebooks/` directory:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 📈 Results

### Performance Comparison

| Model | MAE ↓ | MSE ↓ | R² ↑ |
|-------|-------|-------|------|
| Autoencoder | 0.0100 | 0.0004 | 0.5100 |
| MLP | 0.0100 | 0.1000 | 0.8800 |
| RF + Autoencoder | 0.1730 | 0.0670 | 0.9200 |
| **MLP + Autoencoder** | **0.1059** | **0.0310** | **0.9600** |
| TabNet | 0.0316 | 0.1777 | 0.9600 |
| Wide & Deep | 0.0150 | 0.0066 | 0.9150 |

### Key Findings

🏆 **Best Model**: MLP + Autoencoder
- Highest R² score (0.96)
- Balanced MAE and MSE
- Stable training convergence
- Good generalization on test data

### Visualizations

<div align="center">
  <img src="docs/images/learning_curves.png" width="45%" />
  <img src="docs/images/predictions_scatter.png" width="45%" />
</div>



## 🔬 Methodology

### Feature Engineering

The MOS score is estimated using a weighted combination:

```
MOS = f(0.4 × Duration + 0.3 × Pages + 0.3 × (1 - Bounce Rate))
```

Features are transformed using:
- Log transformation: `log1p(x)`
- Min-Max normalization: `(x - min) / (max - min)`
- Percentile-based scaling

### Evaluation Metrics

- **MAE (Mean Absolute Error)**: Average magnitude of errors
- **MSE (Mean Squared Error)**: Penalizes large errors
- **RMSE (Root Mean Squared Error)**: Same units as target variable
- **R² (Coefficient of Determination)**: Proportion of variance explained



## 🙏 Acknowledgments

**Authors:**
- Mohamed Rayen Mettali
- Ahmed Khalil Sghaier
- Helmi Bouhlel

**Supervisor:**
- Mme Nawres Abdelwahed

**Institution:**
- École Nationale d'Ingénieurs de Carthage (ENICarthage)
- Université de Carthage
- Academic Year 2024-2025

### References

1. Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill.
2. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.
3. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
4. Arik, S. O., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. *AAAI*.
5. Cheng, H. T., et al. (2016). Wide & deep learning for recommender systems. *ACM*.

## 📧 Contact

For questions or collaboration opportunities:

- Email: [rayenmettali1@gmail.com](mailto:rayenmettali1@gmail.com)

---

<div align="center">
  <p>Made with ❤️ for better User Experience</p>
  <p>⭐ Star this repo if you find it helpful!</p>
</div>
