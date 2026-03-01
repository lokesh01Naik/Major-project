#!/bin/bash

# ESG Investment Analytics - Setup Script
# Automated project setup for Unix/Linux/Mac systems

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  ESG Investment Analytics - Automated Setup                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check Python installation
echo "🔍 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Python $PYTHON_VERSION found"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip -q
echo "✅ pip upgraded"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt -q
echo "✅ All dependencies installed"
echo ""

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/models
mkdir -p outputs
mkdir -p notebooks
echo "✅ Directory structure created"
echo ""

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/models/.gitkeep
touch outputs/.gitkeep
echo "✅ Git placeholder files created"
echo ""

# Verify installation
echo "🧪 Verifying installation..."
python3 << END
try:
    import pandas
    import numpy
    import sklearn
    import streamlit
    import plotly
    import matplotlib
    import seaborn
    print("✅ All packages verified successfully!")
except ImportError as e:
    print(f"❌ Package verification failed: {e}")
    exit(1)
END
echo ""

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  ✨ Setup Complete!                                          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Next Steps:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Download datasets from Kaggle (optional):"
echo "     - ESG: https://www.kaggle.com/datasets/debashis74017/esg-scores-and-ratings"
echo "     - Place in: data/raw/"
echo ""
echo "  3. Run the complete pipeline:"
echo "     python main.py"
echo ""
echo "  4. Or launch the dashboard:"
echo "     streamlit run dashboard.py"
echo ""
echo "🎉 Happy analyzing!"
