"""
Machine Learning Models Module
Implements predictive models for ESG investment analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


class ESGInvestmentModels:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
    
    def prepare_data(self, df, target_column='Annual_Return', test_size=0.2, random_state=42):
        """
        Prepare data for modeling
        
        Args:
            df: Input dataframe
            target_column: Target variable for prediction
            test_size: Proportion of test set
            random_state: Random seed
        """
        print("Preparing data for modeling...")
        
        # Select features for modeling
        feature_columns = [
            'ESG_Score', 'Environmental_Score', 'Social_Score', 'Governance_Score',
            'Market_Cap', 'Revenue', 'Profit_Margin', 'PE_Ratio', 'PB_Ratio',
            'Dividend_Yield', 'Beta'
        ]
        
        # Filter columns that exist in the dataframe
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Remove rows with missing target values
        df_model = df.dropna(subset=[target_column])
        
        # Prepare features and target
        X = df_model[available_features].fillna(df_model[available_features].median())
        y = df_model[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Features used: {len(available_features)}")
        
        return X_train, X_test, y_train, y_test, available_features
    
    def train_models(self, X_train, y_train):
        """
        Train multiple regression models
        """
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Define models
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        }
        
        # Train each model
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            try:
                model.fit(X_train, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                           scoring='r2', n_jobs=-1)
                
                self.models[name] = {
                    'model': model,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"  Cross-validation R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
        
        print("\nAll models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models on test set
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        results = []
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R² Score': r2,
                'CV Mean': model_info['cv_mean'],
                'CV Std': model_info['cv_std']
            })
            
            print(f"\n{name}:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R² Score: {r2:.4f}")
        
        # Create results dataframe
        results_df = pd.DataFrame(results).sort_values('R² Score', ascending=False)
        
        # Identify best model
        best_model_idx = results_df['R² Score'].idxmax()
        self.best_model_name = results_df.loc[best_model_idx, 'Model']
        self.best_model = self.models[self.best_model_name]['model']
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"R² Score: {results_df.loc[best_model_idx, 'R² Score']:.4f}")
        print(f"{'='*60}")
        
        return results_df
    
    def feature_importance_analysis(self, feature_names, output_dir='outputs'):
        """
        Analyze and visualize feature importance
        """
        print("\nAnalyzing feature importance...")
        
        if self.best_model is None:
            print("No best model found. Please train and evaluate models first.")
            return None
        
        # Get feature importance (only works for tree-based models)
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            self.feature_importance = importance_df
            
            # Visualize
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
            plt.title(f'Feature Importance - {self.best_model_name}', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.tight_layout()
            
            # Save
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {output_path}")
            plt.close()
            
            return importance_df
        else:
            print(f"{self.best_model_name} does not support feature importance.")
            return None
    
    def predict_returns(self, X):
        """
        Make predictions using the best model
        """
        if self.best_model is None:
            raise ValueError("No model trained. Please train models first.")
        
        predictions = self.best_model.predict(X)
        return predictions
    
    def create_investment_score(self, df, esg_weight=0.4, financial_weight=0.6):
        """
        Create comprehensive investment score combining ESG and predicted returns
        
        Args:
            df: Dataframe with ESG scores and features
            esg_weight: Weight for ESG score (default 40%)
            financial_weight: Weight for predicted financial performance (default 60%)
        """
        print("\nCreating investment scores...")
        
        df_scored = df.copy()
        
        # Normalize ESG score to 0-100 range
        esg_normalized = df_scored['ESG_Score'].fillna(df_scored['ESG_Score'].median())
        
        # Predict returns using the model
        feature_columns = [
            'ESG_Score', 'Environmental_Score', 'Social_Score', 'Governance_Score',
            'Market_Cap', 'Revenue', 'Profit_Margin', 'PE_Ratio', 'PB_Ratio',
            'Dividend_Yield', 'Beta'
        ]
        available_features = [col for col in feature_columns if col in df_scored.columns]
        X = df_scored[available_features].fillna(df_scored[available_features].median())
        
        predicted_returns = self.predict_returns(X)
        
        # Normalize predicted returns to 0-100 range
        returns_min, returns_max = predicted_returns.min(), predicted_returns.max()
        returns_normalized = ((predicted_returns - returns_min) / (returns_max - returns_min + 0.0001)) * 100
        
        # Calculate investment score
        df_scored['Predicted_Return'] = predicted_returns
        df_scored['Investment_Score'] = (
            esg_normalized * esg_weight + 
            returns_normalized * financial_weight
        )
        
        # Add investment recommendation
        df_scored['Recommendation'] = pd.cut(
            df_scored['Investment_Score'],
            bins=[0, 40, 60, 80, 100],
            labels=['Avoid', 'Hold', 'Buy', 'Strong Buy']
        )
        
        print(f"Investment scores created for {len(df_scored)} companies")
        
        return df_scored
    
    def save_models(self, output_dir='data/models'):
        """
        Save trained models to disk
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model_info in self.models.items():
            model_filename = name.lower().replace(' ', '_') + '.pkl'
            model_path = os.path.join(output_dir, model_filename)
            joblib.dump(model_info['model'], model_path)
            print(f"Saved {name} to {model_path}")
        
        # Save best model separately
        if self.best_model is not None:
            best_model_path = os.path.join(output_dir, 'best_model.pkl')
            joblib.dump(self.best_model, best_model_path)
            print(f"\nBest model ({self.best_model_name}) saved to {best_model_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model from disk
        """
        self.best_model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return self.best_model


if __name__ == "__main__":
    # Example usage
    print("ESG Investment Models - Example Usage\n")
    
    # Load preprocessed data
    df = pd.read_csv('data/processed/cleaned_esg_financial.csv')
    
    # Initialize model class
    esg_models = ESGInvestmentModels()
    
    # Prepare data
    X_train, X_test, y_train, y_test, features = esg_models.prepare_data(df)
    
    # Train models
    esg_models.train_models(X_train, y_train)
    
    # Evaluate models
    results = esg_models.evaluate_models(X_test, y_test)
    print("\n\nModel Comparison:")
    print(results.to_string(index=False))
    
    # Feature importance
    importance = esg_models.feature_importance_analysis(features)
    if importance is not None:
        print("\n\nTop 5 Important Features:")
        print(importance.head().to_string(index=False))
    
    # Create investment scores
    scored_df = esg_models.create_investment_score(df)
    print("\n\nTop 10 Investment Recommendations:")
    print(scored_df[['Ticker', 'Company', 'ESG_Score', 'Predicted_Return', 
                     'Investment_Score', 'Recommendation']]
          .sort_values('Investment_Score', ascending=False)
          .head(10)
          .to_string(index=False))
    
    # Save models
    esg_models.save_models()
    
    print("\n" + "="*60)
    print("MODELING COMPLETE!")
    print("="*60)
