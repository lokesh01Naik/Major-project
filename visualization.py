"""
Visualization Module
Creates charts and visualizations for ESG investment analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


class ESGVisualizer:
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """Initialize visualizer with default style"""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Set color palette
        self.colors = {
            'primary': '#2E5090',
            'secondary': '#4A90E2',
            'success': '#27AE60',
            'warning': '#F39C12',
            'danger': '#E74C3C',
            'info': '#3498DB'
        }
        
        sns.set_palette("husl")
    
    def plot_esg_distribution(self, df, output_dir='outputs'):
        """Plot ESG score distribution by sector"""
        print("Creating ESG distribution plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ESG Scores Distribution Analysis', fontsize=18, fontweight='bold', y=0.995)
        
        # Overall ESG Score Distribution
        axes[0, 0].hist(df['ESG_Score'], bins=30, color=self.colors['primary'], 
                       edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Overall ESG Score Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('ESG Score', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].axvline(df['ESG_Score'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {df["ESG_Score"].mean():.2f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ESG Score by Sector
        if 'Sector' in df.columns:
            sector_data = df.groupby('Sector')['ESG_Score'].apply(list)
            axes[0, 1].boxplot(sector_data.values, labels=sector_data.index)
            axes[0, 1].set_title('ESG Score Distribution by Sector', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Sector', fontsize=12)
            axes[0, 1].set_ylabel('ESG Score', fontsize=12)
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # ESG Components Distribution
        if all(col in df.columns for col in ['Environmental_Score', 'Social_Score', 'Governance_Score']):
            components = ['Environmental_Score', 'Social_Score', 'Governance_Score']
            component_means = [df[col].mean() for col in components]
            colors_list = [self.colors['success'], self.colors['info'], self.colors['warning']]
            
            axes[1, 0].bar(range(len(components)), component_means, color=colors_list, 
                          edgecolor='black', alpha=0.7)
            axes[1, 0].set_title('Average ESG Component Scores', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Component', fontsize=12)
            axes[1, 0].set_ylabel('Average Score', fontsize=12)
            axes[1, 0].set_xticks(range(len(components)))
            axes[1, 0].set_xticklabels(['Environmental', 'Social', 'Governance'], rotation=45)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            for i, v in enumerate(component_means):
                axes[1, 0].text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')
        
        # ESG Risk Distribution
        if 'ESG_Risk' in df.columns:
            risk_counts = df['ESG_Risk'].value_counts()
            axes[1, 1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                          colors=[self.colors['success'], self.colors['warning'], self.colors['danger']],
                          startangle=90)
            axes[1, 1].set_title('ESG Risk Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'esg_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ESG distribution plot saved to {output_path}")
        plt.close()
    
    def plot_esg_vs_performance(self, df, output_dir='outputs'):
        """Plot ESG scores vs financial performance"""
        print("Creating ESG vs Performance plot...")
        
        if 'Annual_Return' not in df.columns:
            print("Annual_Return column not found. Skipping this visualization.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('ESG Performance vs Financial Returns', fontsize=18, fontweight='bold')
        
        # Scatter plot: ESG Score vs Annual Return
        scatter = axes[0].scatter(df['ESG_Score'], df['Annual_Return'], 
                                 c=df['ESG_Score'], cmap='viridis', 
                                 s=100, alpha=0.6, edgecolors='black')
        axes[0].set_title('ESG Score vs Annual Return', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('ESG Score', fontsize=12)
        axes[0].set_ylabel('Annual Return (%)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['ESG_Score'].dropna(), df['Annual_Return'].dropna(), 1)
        p = np.poly1d(z)
        axes[0].plot(df['ESG_Score'], p(df['ESG_Score']), "r--", linewidth=2, 
                    label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        axes[0].legend()
        plt.colorbar(scatter, ax=axes[0], label='ESG Score')
        
        # Average returns by ESG category
        if 'ESG_Category' in df.columns:
            category_returns = df.groupby('ESG_Category')['Annual_Return'].mean().sort_values()
            axes[1].barh(range(len(category_returns)), category_returns.values, 
                        color=self.colors['secondary'], edgecolor='black', alpha=0.7)
            axes[1].set_yticks(range(len(category_returns)))
            axes[1].set_yticklabels(category_returns.index)
            axes[1].set_title('Average Returns by ESG Category', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Average Annual Return (%)', fontsize=12)
            axes[1].grid(True, alpha=0.3, axis='x')
            
            for i, v in enumerate(category_returns.values):
                axes[1].text(v + 0.5, i, f'{v:.2f}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'esg_vs_performance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ESG vs Performance plot saved to {output_path}")
        plt.close()
    
    def create_sector_analysis(self, df, output_dir='outputs'):
        """Create sector-wise ESG analysis"""
        print("Creating sector analysis...")
        
        if 'Sector' not in df.columns:
            print("Sector column not found. Skipping this visualization.")
            return
        
        # Calculate sector statistics
        sector_stats = df.groupby('Sector').agg({
            'ESG_Score': 'mean',
            'Environmental_Score': 'mean',
            'Social_Score': 'mean',
            'Governance_Score': 'mean',
            'Ticker': 'count'
        }).round(2)
        sector_stats.columns = ['ESG_Score', 'Environmental', 'Social', 'Governance', 'Count']
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Sector-wise ESG Analysis', fontsize=18, fontweight='bold')
        
        # Grouped bar chart
        sector_stats[['Environmental', 'Social', 'Governance']].plot(
            kind='bar', ax=axes[0], color=[self.colors['success'], 
            self.colors['info'], self.colors['warning']], 
            edgecolor='black', alpha=0.7
        )
        axes[0].set_title('ESG Component Scores by Sector', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Sector', fontsize=12)
        axes[0].set_ylabel('Average Score', fontsize=12)
        axes[0].legend(title='Component', loc='upper right')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Overall ESG score by sector
        sector_stats['ESG_Score'].plot(kind='barh', ax=axes[1], 
                                       color=self.colors['primary'], 
                                       edgecolor='black', alpha=0.7)
        axes[1].set_title('Average ESG Score by Sector', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('ESG Score', fontsize=12)
        axes[1].set_ylabel('Sector', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        for i, v in enumerate(sector_stats['ESG_Score'].values):
            axes[1].text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'sector_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Sector analysis saved to {output_path}")
        plt.close()
        
        return sector_stats
    
    def create_interactive_dashboard(self, df, output_dir='outputs'):
        """Create interactive Plotly dashboard"""
        print("Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ESG Score Distribution', 'ESG vs Annual Return',
                          'Sector Comparison', 'Risk Distribution'),
            specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        # ESG Score Distribution
        fig.add_trace(
            go.Histogram(x=df['ESG_Score'], name='ESG Score', 
                        marker_color=self.colors['primary']),
            row=1, col=1
        )
        
        # ESG vs Annual Return
        if 'Annual_Return' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['ESG_Score'], y=df['Annual_Return'], 
                          mode='markers', name='Companies',
                          marker=dict(size=10, color=df['ESG_Score'], 
                                    colorscale='Viridis', showscale=True)),
                row=1, col=2
            )
        
        # Sector Comparison
        if 'Sector' in df.columns:
            sector_avg = df.groupby('Sector')['ESG_Score'].mean().sort_values()
            fig.add_trace(
                go.Bar(x=sector_avg.values, y=sector_avg.index, 
                      orientation='h', name='Avg ESG Score',
                      marker_color=self.colors['secondary']),
                row=2, col=1
            )
        
        # Risk Distribution
        if 'ESG_Risk' in df.columns:
            risk_counts = df['ESG_Risk'].value_counts()
            fig.add_trace(
                go.Pie(labels=risk_counts.index, values=risk_counts.values, 
                      name='ESG Risk'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="ESG Investment Analytics Dashboard",
            title_font_size=20,
            showlegend=True,
            height=800
        )
        
        # Save
        output_path = os.path.join(output_dir, 'interactive_dashboard.html')
        fig.write_html(output_path)
        print(f"Interactive dashboard saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    visualizer = ESGVisualizer()
    
    # Load data
    df = pd.read_csv('data/processed/cleaned_esg_financial.csv')
    
    # Create all visualizations
    visualizer.plot_esg_distribution(df)
    visualizer.plot_esg_vs_performance(df)
    visualizer.create_sector_analysis(df)
    visualizer.create_interactive_dashboard(df)
    
    print("\nAll visualizations created successfully!")
