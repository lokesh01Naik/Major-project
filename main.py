"""
Main Execution Script
Complete end-to-end pipeline for ESG Investment Analytics
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from preprocessing import DataPreprocessor
from models import ESGInvestmentModels
from visualization import ESGVisualizer

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        'data',
        'data/raw',
        'data/processed',
        'data/models',
        'outputs',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directory structure created")


def run_data_collection():
    """Phase 1: Data Collection"""
    print("\n" + "="*70)
    print("PHASE 1: DATA COLLECTION")
    print("="*70)
    
    loader = DataLoader()
    
    # Load or generate ESG data
    esg_df = loader.load_esg_data()
    print(f"ESG Data Shape: {esg_df.shape}")
    
    # Download financial data for sample tickers
    tickers = esg_df['Ticker'].tolist()[:10]  # Limit for demo
    print(f"\nDownloading financial data for {len(tickers)} companies...")
    financial_data = loader.download_financial_data(tickers)
    
    # Create merged dataset
    merged_df = loader.create_merged_dataset(esg_df, financial_data)
    print(f"Merged Dataset Shape: {merged_df.shape}")
    
    return merged_df


def run_preprocessing(df):
    """Phase 2: Data Preprocessing"""
    print("\n" + "="*70)
    print("PHASE 2: DATA PREPROCESSING")
    print("="*70)
    
    preprocessor = DataPreprocessor()
    
    # Data quality check
    preprocessor.check_data_quality(df)
    
    # Handle missing values
    df_clean = preprocessor.handle_missing_values(df, strategy='median')
    
    # Create ESG features
    df_featured = preprocessor.create_esg_features(df_clean)
    
    # Generate correlation matrix
    preprocessor.generate_correlation_matrix(df_featured)
    
    # Save preprocessed data
    output_path = 'data/processed/cleaned_esg_financial.csv'
    preprocessor.save_preprocessed_data(df_featured, output_path)
    
    return df_featured


def run_modeling(df):
    """Phase 3: Predictive Modeling"""
    print("\n" + "="*70)
    print("PHASE 3: PREDICTIVE MODELING")
    print("="*70)
    
    esg_models = ESGInvestmentModels()
    
    # Prepare data
    X_train, X_test, y_train, y_test, features = esg_models.prepare_data(df)
    
    # Train models
    esg_models.train_models(X_train, y_train)
    
    # Evaluate models
    results = esg_models.evaluate_models(X_test, y_test)
    print("\n📊 Model Performance Comparison:")
    print(results.to_string(index=False))
    
    # Feature importance
    importance = esg_models.feature_importance_analysis(features)
    if importance is not None:
        print("\n🎯 Top 5 Important Features:")
        print(importance.head().to_string(index=False))
    
    # Create investment scores
    scored_df = esg_models.create_investment_score(df)
    
    # Save models
    esg_models.save_models()
    
    # Save scored data
    scored_df.to_csv('data/processed/investment_scores.csv', index=False)
    print("\n✅ Investment scores saved")
    
    return scored_df, results


def run_visualization(df):
    """Phase 4: Visualization"""
    print("\n" + "="*70)
    print("PHASE 4: VISUALIZATION")
    print("="*70)
    
    visualizer = ESGVisualizer()
    
    # Create all visualizations
    visualizer.plot_esg_distribution(df)
    visualizer.plot_esg_vs_performance(df)
    sector_stats = visualizer.create_sector_analysis(df)
    visualizer.create_interactive_dashboard(df)
    
    if sector_stats is not None:
        print("\n📊 Sector Statistics:")
        print(sector_stats)
    
    print("\n✅ All visualizations created in 'outputs/' directory")


def generate_report(scored_df, model_results):
    """Generate final report"""
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    
    print(f"\n📈 Dataset Summary:")
    print(f"  Total Companies Analyzed: {len(scored_df)}")
    print(f"  Average ESG Score: {scored_df['ESG_Score'].mean():.2f}")
    print(f"  Average Predicted Return: {scored_df['Predicted_Return'].mean():.2f}%")
    
    print(f"\n🏆 Best Model: {model_results.iloc[0]['Model']}")
    print(f"  R² Score: {model_results.iloc[0]['R² Score']:.4f}")
    print(f"  RMSE: {model_results.iloc[0]['RMSE']:.4f}")
    
    print(f"\n💼 Investment Recommendations:")
    rec_counts = scored_df['Recommendation'].value_counts()
    for rec, count in rec_counts.items():
        print(f"  {rec}: {count} companies ({count/len(scored_df)*100:.1f}%)")
    
    print(f"\n🌟 Top 5 Investment Opportunities:")
    top_5 = scored_df.nlargest(5, 'Investment_Score')[
        ['Ticker', 'Company', 'ESG_Score', 'Predicted_Return', 'Investment_Score', 'Recommendation']
    ]
    print(top_5.to_string(index=False))
    
    # Save summary report
    with open('outputs/summary_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("ESG INVESTMENT ANALYTICS - FINAL REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total Companies Analyzed: {len(scored_df)}\n")
        f.write(f"Average ESG Score: {scored_df['ESG_Score'].mean():.2f}\n")
        f.write(f"Average Predicted Return: {scored_df['Predicted_Return'].mean():.2f}%\n\n")
        f.write(f"Best Model: {model_results.iloc[0]['Model']}\n")
        f.write(f"R² Score: {model_results.iloc[0]['R² Score']:.4f}\n\n")
        f.write("Investment Recommendations:\n")
        for rec, count in rec_counts.items():
            f.write(f"  {rec}: {count} companies\n")
        f.write("\n" + top_5.to_string(index=False))
    
    print("\n✅ Summary report saved to 'outputs/summary_report.txt'")


def main():
    """Main execution function"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  ESG INVESTMENT ANALYTICS - COMPLETE PIPELINE                ║
    ║  Business Analytics for Green Finance                        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Step 0: Setup
        create_directory_structure()
        
        # Step 1: Data Collection
        merged_df = run_data_collection()
        
        # Step 2: Preprocessing
        cleaned_df = run_preprocessing(merged_df)
        
        # Step 3: Modeling
        scored_df, model_results = run_modeling(cleaned_df)
        
        # Step 4: Visualization
        run_visualization(scored_df)
        
        # Step 5: Generate Report
        generate_report(scored_df, model_results)
        
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        print("\n📁 Output Files:")
        print("  • Data: data/processed/")
        print("  • Models: data/models/")
        print("  • Visualizations: outputs/")
        print("  • Report: outputs/summary_report.txt")
        
        print("\n🚀 Next Steps:")
        print("  1. Review visualizations in the 'outputs/' directory")
        print("  2. Check the summary report: outputs/summary_report.txt")
        print("  3. Run the dashboard: streamlit run dashboard.py")
        print("  4. Explore data in notebooks/")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
