"""
Data Loader Module
Downloads ESG and financial data from various sources
"""

import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import os

class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def load_esg_data(self, filepath=None):
        """
        Load ESG dataset from local file or generate sample data
        
        Dataset sources:
        1. Kaggle: https://www.kaggle.com/datasets/debashis74017/esg-scores-and-ratings
        2. Refinitiv: https://www.kaggle.com/datasets/pritish509/refinitiv-esg-dataset
        """
        if filepath and os.path.exists(filepath):
            print(f"Loading ESG data from {filepath}")
            df = pd.read_csv(filepath)
        else:
            print("Generating sample ESG data...")
            df = self._generate_sample_esg_data()
        
        return df
    
    def _generate_sample_esg_data(self):
        """Generate sample ESG data for testing"""
        import numpy as np
        
        companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 
                    'JPM', 'V', 'WMT', 'PG', 'JNJ', 'UNH', 'HD', 'MA',
                    'BAC', 'XOM', 'CVX', 'PFE', 'KO', 'PEP', 'COST', 
                    'ABBV', 'TMO', 'NKE', 'DIS', 'CSCO', 'ADBE', 'NFLX', 'CRM']
        
        sectors = ['Technology', 'Finance', 'Healthcare', 'Consumer', 'Energy']
        
        data = {
            'Ticker': companies,
            'Company': [f'Company {ticker}' for ticker in companies],
            'Sector': np.random.choice(sectors, len(companies)),
            'ESG_Score': np.random.uniform(40, 95, len(companies)),
            'Environmental_Score': np.random.uniform(30, 100, len(companies)),
            'Social_Score': np.random.uniform(30, 100, len(companies)),
            'Governance_Score': np.random.uniform(30, 100, len(companies)),
            'ESG_Risk': np.random.choice(['Low', 'Medium', 'High'], len(companies)),
            'Year': [2023] * len(companies)
        }
        
        df = pd.DataFrame(data)
        
        # Save sample data
        output_path = os.path.join(self.raw_dir, 'sample_esg_data.csv')
        df.to_csv(output_path, index=False)
        print(f"Sample ESG data saved to {output_path}")
        
        return df
    
    def download_financial_data(self, tickers, start_date=None, end_date=None):
        """
        Download financial data from Yahoo Finance
        
        Args:
            tickers: List of stock tickers
            start_date: Start date (default: 1 year ago)
            end_date: End date (default: today)
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        financial_data = {}
        
        print(f"Downloading financial data for {len(tickers)} tickers...")
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                
                # Get historical price data
                hist = stock.history(start=start_date, end=end_date)
                
                # Get fundamental data
                info = stock.info
                
                financial_data[ticker] = {
                    'price_history': hist,
                    'market_cap': info.get('marketCap', None),
                    'revenue': info.get('totalRevenue', None),
                    'profit_margin': info.get('profitMargins', None),
                    'pe_ratio': info.get('trailingPE', None),
                    'pb_ratio': info.get('priceToBook', None),
                    'dividend_yield': info.get('dividendYield', None),
                    'beta': info.get('beta', None),
                    'sector': info.get('sector', 'Unknown')
                }
                
                print(f"  ✓ Downloaded data for {ticker}")
                
            except Exception as e:
                print(f"  ✗ Error downloading {ticker}: {str(e)}")
                financial_data[ticker] = None
        
        return financial_data
    
    def create_merged_dataset(self, esg_df, financial_data):
        """
        Merge ESG data with financial data
        """
        merged_data = []
        
        for _, row in esg_df.iterrows():
            ticker = row['Ticker']
            
            if ticker in financial_data and financial_data[ticker] is not None:
                fin_data = financial_data[ticker]
                
                # Calculate returns
                price_hist = fin_data['price_history']
                if not price_hist.empty:
                    annual_return = ((price_hist['Close'][-1] / price_hist['Close'][0]) - 1) * 100
                else:
                    annual_return = None
                
                merged_row = {
                    'Ticker': ticker,
                    'Company': row['Company'],
                    'Sector': row['Sector'],
                    'ESG_Score': row['ESG_Score'],
                    'Environmental_Score': row['Environmental_Score'],
                    'Social_Score': row['Social_Score'],
                    'Governance_Score': row['Governance_Score'],
                    'ESG_Risk': row['ESG_Risk'],
                    'Market_Cap': fin_data.get('market_cap'),
                    'Revenue': fin_data.get('revenue'),
                    'Profit_Margin': fin_data.get('profit_margin'),
                    'PE_Ratio': fin_data.get('pe_ratio'),
                    'PB_Ratio': fin_data.get('pb_ratio'),
                    'Dividend_Yield': fin_data.get('dividend_yield'),
                    'Beta': fin_data.get('beta'),
                    'Annual_Return': annual_return
                }
                
                merged_data.append(merged_row)
        
        merged_df = pd.DataFrame(merged_data)
        
        # Save merged dataset
        output_path = os.path.join(self.processed_dir, 'merged_esg_financial.csv')
        merged_df.to_csv(output_path, index=False)
        print(f"\nMerged dataset saved to {output_path}")
        
        return merged_df


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load ESG data
    esg_df = loader.load_esg_data()
    print(f"\nLoaded ESG data: {esg_df.shape}")
    print(esg_df.head())
    
    # Download financial data
    tickers = esg_df['Ticker'].tolist()[:10]  # Limit to first 10 for demo
    financial_data = loader.download_financial_data(tickers)
    
    # Create merged dataset
    merged_df = loader.create_merged_dataset(esg_df, financial_data)
    print(f"\nMerged dataset: {merged_df.shape}")
    print(merged_df.head())
