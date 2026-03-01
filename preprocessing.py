"""
Data Preprocessing Module
Handles data cleaning, transformation, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
    
    def check_data_quality(self, df):
        """
        Generate data quality report
        """
        print("=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        
        print(f"\nDataset Shape: {df.shape}")
        print(f"Number of Records: {df.shape[0]}")
        print(f"Number of Features: {df.shape[1]}")
        
        print("\n--- Missing Values ---")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False))
        
        print("\n--- Data Types ---")
        print(df.dtypes.value_counts())
        
        print("\n--- Numerical Features Statistics ---")
        print(df.describe())
        
        print("\n--- Categorical Features ---")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"\n{col}: {df[col].nunique()} unique values")
            print(df[col].value_counts().head())
        
        return missing_df
    
    def handle_missing_values(self, df, strategy='median'):
        """
        Handle missing values in the dataset
        
        Args:
            df: Input dataframe
            strategy: 'median', 'mean', or 'drop'
        """
        print("\nHandling missing values...")
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if strategy == 'drop':
            # Drop rows with any missing values
            df_clean = df.dropna()
            print(f"Dropped {len(df) - len(df_clean)} rows with missing values")
        else:
            df_clean = df.copy()
            
            # Handle numerical columns
            if len(numerical_cols) > 0:
                if strategy == 'median':
                    imputer = SimpleImputer(strategy='median')
                elif strategy == 'mean':
                    imputer = SimpleImputer(strategy='mean')
                
                df_clean[numerical_cols] = imputer.fit_transform(df_clean[numerical_cols])
            
            # Handle categorical columns - fill with mode
            for col in categorical_cols:
                if df_clean[col].isnull().any():
                    mode_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                    df_clean[col].fillna(mode_value, inplace=True)
        
        print(f"Missing values remaining: {df_clean.isnull().sum().sum()}")
        
        return df_clean
    
    def remove_outliers(self, df, columns, method='iqr', threshold=1.5):
        """
        Remove outliers from specified columns
        
        Args:
            df: Input dataframe
            columns: List of columns to check for outliers
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or Z-score threshold
        """
        print(f"\nRemoving outliers using {method} method...")
        df_clean = df.copy()
        initial_count = len(df_clean)
        
        for col in columns:
            if col not in df_clean.columns or not np.issubdtype(df_clean[col].dtype, np.number):
                continue
            
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                df_clean = df_clean[z_scores < threshold]
        
        removed_count = initial_count - len(df_clean)
        print(f"Removed {removed_count} outlier records ({removed_count/initial_count*100:.2f}%)")
        
        return df_clean
    
    def create_esg_features(self, df):
        """
        Create additional ESG-related features
        """
        print("\nCreating ESG features...")
        
        df_featured = df.copy()
        
        # Composite ESG score (if individual scores exist)
        if all(col in df.columns for col in ['Environmental_Score', 'Social_Score', 'Governance_Score']):
            df_featured['ESG_Composite'] = (
                df_featured['Environmental_Score'] * 0.33 +
                df_featured['Social_Score'] * 0.33 +
                df_featured['Governance_Score'] * 0.34
            )
        
        # ESG risk category to numerical
        if 'ESG_Risk' in df.columns:
            risk_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
            df_featured['ESG_Risk_Numerical'] = df_featured['ESG_Risk'].map(risk_mapping)
        
        # ESG score categories
        if 'ESG_Score' in df.columns:
            df_featured['ESG_Category'] = pd.cut(
                df_featured['ESG_Score'],
                bins=[0, 50, 70, 85, 100],
                labels=['Poor', 'Fair', 'Good', 'Excellent']
            )
        
        # Performance score (combining ESG and financial metrics)
        if 'Annual_Return' in df.columns and 'ESG_Score' in df.columns:
            # Normalize scores to 0-1 range for fair combination
            esg_normalized = (df_featured['ESG_Score'] - df_featured['ESG_Score'].min()) / \
                           (df_featured['ESG_Score'].max() - df_featured['ESG_Score'].min())
            
            # Handle potential NaN in returns
            returns_clean = df_featured['Annual_Return'].fillna(0)
            returns_normalized = (returns_clean - returns_clean.min()) / \
                                (returns_clean.max() - returns_clean.min() + 0.0001)
            
            df_featured['Performance_Score'] = (esg_normalized * 0.4 + returns_normalized * 0.6) * 100
        
        print(f"Created {len(df_featured.columns) - len(df.columns)} new features")
        
        return df_featured
    
    def encode_categorical_variables(self, df, columns=None):
        """
        Encode categorical variables
        """
        print("\nEncoding categorical variables...")
        
        df_encoded = df.copy()
        
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[f'{col}_Encoded'] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print(f"  Encoded {col}: {df_encoded[col].nunique()} categories")
        
        return df_encoded
    
    def normalize_features(self, df, columns=None):
        """
        Normalize numerical features using StandardScaler
        """
        print("\nNormalizing features...")
        
        df_normalized = df.copy()
        
        if columns is None:
            columns = df_normalized.select_dtypes(include=[np.number]).columns
        
        # Filter out columns that don't exist or have constant values
        valid_columns = []
        for col in columns:
            if col in df_normalized.columns and df_normalized[col].std() > 0:
                valid_columns.append(col)
        
        if valid_columns:
            df_normalized[valid_columns] = self.scaler.fit_transform(df_normalized[valid_columns])
            print(f"Normalized {len(valid_columns)} features")
        
        return df_normalized
    
    def generate_correlation_matrix(self, df, output_dir='outputs'):
        """
        Generate and save correlation matrix visualization
        """
        print("\nGenerating correlation matrix...")
        
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.shape[1] < 2:
            print("Not enough numerical features for correlation analysis")
            return None
        
        # Calculate correlation matrix
        corr_matrix = numerical_df.corr()
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'correlation_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to {output_path}")
        plt.close()
        
        return corr_matrix
    
    def save_preprocessed_data(self, df, output_path):
        """
        Save preprocessed data to CSV
        """
        df.to_csv(output_path, index=False)
        print(f"\nPreprocessed data saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Load sample data
    df = pd.read_csv('data/processed/merged_esg_financial.csv')
    
    # Data quality check
    preprocessor.check_data_quality(df)
    
    # Handle missing values
    df_clean = preprocessor.handle_missing_values(df, strategy='median')
    
    # Create features
    df_featured = preprocessor.create_esg_features(df_clean)
    
    # Generate correlation matrix
    corr_matrix = preprocessor.generate_correlation_matrix(df_featured)
    
    # Save processed data
    preprocessor.save_preprocessed_data(df_featured, 'data/processed/cleaned_esg_financial.csv')
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
