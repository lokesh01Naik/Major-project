# 📊 Business Analytics for Green Finance: ESG Investment Insights for Sustainable Growth

A complete end-to-end pipeline for analyzing ESG (Environmental, Social, and Governance) scores alongside financial performance data to generate data-driven investment insights that support sustainable and responsible growth.

---

## 🌟 Features

- **Automated Data Collection** — Downloads financial data via Yahoo Finance and loads ESG scores from CSV or generates sample data
- **Data Preprocessing** — Missing value handling, outlier removal, feature engineering, and normalization
- **Predictive Modeling** — Trains and compares 5 ML models (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting)
- **Investment Scoring** — Combines ESG scores and predicted returns into a composite investment score
- **Visualizations** — Static charts (matplotlib/seaborn) and an interactive HTML dashboard (Plotly)
- **Streamlit Dashboard** — Live interactive web app for exploring results

---

## 🗂️ Project Structure

```
esg-investment-analytics/
├── main.py               # End-to-end pipeline runner
├── data_loader.py        # Data ingestion and merging
├── preprocessing.py      # Cleaning, feature engineering
├── models.py             # ML model training and evaluation
├── visualization.py      # Charts and interactive dashboard
├── dashboard.py          # Streamlit web app
├── requirements.txt      # Python dependencies
├── setup.sh              # Unix/macOS setup script
├── setup.bat             # Windows setup script
├── data/
│   ├── raw/              # Source data files
│   ├── processed/        # Cleaned and merged datasets
│   └── models/           # Saved model artifacts (.pkl)
├── outputs/              # Charts, HTML dashboard, reports
└── notebooks/            # Jupyter notebooks for exploration
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/esg-investment-analytics.git
cd esg-investment-analytics
```

### 2. Run the setup script

**macOS / Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```bat
setup.bat
```

This creates a virtual environment and installs all dependencies automatically.

### 3. (Optional) Add your own ESG data

Download a dataset from one of these sources and place it in `data/raw/`:

- [ESG Scores & Ratings — Kaggle](https://www.kaggle.com/datasets/debashis74017/esg-scores-and-ratings)
- [Refinitiv ESG Dataset — Kaggle](https://www.kaggle.com/datasets/pritish509/refinitiv-esg-dataset)

If no file is provided, the pipeline auto-generates realistic sample data for 30 companies.

### 4. Run the full pipeline

```bash
source venv/bin/activate  # Windows: venv\Scripts\activate
python main.py
```

### 5. Launch the interactive dashboard

```bash
streamlit run dashboard.py
```

---

## 🔧 Manual Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| pandas | 2.0.0 | Data manipulation |
| numpy | 1.24.0 | Numerical computing |
| scikit-learn | 1.3.0 | Machine learning models |
| matplotlib | 3.7.0 | Static visualizations |
| seaborn | 0.12.0 | Statistical plots |
| plotly | 5.14.0 | Interactive dashboard |
| streamlit | 1.22.0 | Web app interface |
| yfinance | 0.2.18 | Yahoo Finance data |
| joblib | 1.3.0 | Model persistence |

---

## 🧠 Models

Five regression models are trained and evaluated using cross-validation:

| Model | Description |
|---|---|
| Linear Regression | Baseline linear model |
| Ridge Regression | L2-regularized linear model |
| Lasso Regression | L1-regularized linear model |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosted ensemble |

The best-performing model (by R² score) is saved to `data/models/best_model.pkl` and used for investment scoring.

---

## 📈 Investment Scoring

Each company receives an **Investment Score** computed as:

```
Investment Score = ESG Score × 0.4 + Predicted Return (normalized) × 0.6
```

Companies are then categorized into recommendations:

| Score Range | Recommendation |
|---|---|
| 80 – 100 | ✅ Strong Buy |
| 60 – 80 | 🟡 Buy |
| 40 – 60 | 🟠 Hold |
| 0 – 40 | 🔴 Avoid |

---

## 📁 Outputs

After running the pipeline, the following files are generated:

```
outputs/
├── esg_distribution.png       # ESG score distributions
├── esg_vs_performance.png     # ESG scores vs financial returns
├── sector_analysis.png        # Sector-level ESG breakdown
├── feature_importance.png     # Model feature importance
├── correlation_matrix.png     # Feature correlation heatmap
├── interactive_dashboard.html # Standalone Plotly dashboard
└── summary_report.txt         # Final text report
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.
