import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from betting_analysis import run_analysis
import os

def generate_results_tables(results):
    """Generate all tables needed for Section 4 (Results)"""
    
    performance = results['performance_summary']
    
    # Table: Strategy Performance Comparison
    comparison_table = pd.DataFrame({
        'Strategy': list(performance.keys()),
        'Total P&L (£)': [perf['metrics']['Total P&L'] for perf in performance.values()],
        'Total Bets': [perf['metrics']['Total Bets'] for perf in performance.values()],
        'Win Rate (%)': [perf['metrics']['Win Rate'] * 100 for perf in performance.values()],
        'ROI (%)': [perf['metrics']['ROI (%)'] for perf in performance.values()],
        'Avg Weekly Return (£)': [perf['metrics']['Mean Weekly Return'] for perf in performance.values()],
        'Std Weekly Return (£)': [perf['metrics']['Std Weekly Return'] for perf in performance.values()],
        'Sharpe Ratio': [perf['metrics']['Sharpe Ratio'] for perf in performance.values()],
        'Sortino Ratio': [perf['metrics']['Sortino Ratio'] for perf in performance.values()],
        'Calmar Ratio': [perf['metrics']['Calmar Ratio'] for perf in performance.values()],
        'Omega Ratio': [perf['metrics']['Omega Ratio'] for perf in performance.values()],
        'Max Drawdown (£)': [perf['metrics']['Max Drawdown'] for perf in performance.values()],
        'VaR 95% (£)': [perf['metrics']['VaR 95%'] for perf in performance.values()],
        'ES 95% (£)': [perf['metrics']['ES 95%'] for perf in performance.values()]
    })
    
    # Table: By-Season Performance
    seasons = ['2019/20', '2020/21', '2021/22', '2022/23', '2023/24']
    season_performance = []
    
    for strategy_name, perf_data in performance.items():
        results_df = perf_data['results']
        results_df['Season'] = pd.to_datetime(results_df['Date']).dt.year
        
        for year in results_df['Season'].unique():
            season_data = results_df[results_df['Season'] == year]
            season_performance.append({
                'Strategy': strategy_name,
                'Season': year,
                'P&L': season_data['PnL'].sum(),
                'Bets': len(season_data),
                'Win Rate': season_data['Win'].mean()
            })
    
    season_table = pd.DataFrame(season_performance)
    
    # Table: Risk Metrics Detail
    risk_table = pd.DataFrame({
        'Strategy': list(performance.keys()),
        'Mean Weekly Return': [perf['metrics']['Mean Weekly Return'] for perf in performance.values()],
        'Std Weekly Return': [perf['metrics']['Std Weekly Return'] for perf in performance.values()],
        'Skewness': [pd.Series(perf['results'].groupby(pd.to_datetime(perf['results']['Date']).dt.to_period('W'))['PnL'].sum()).skew() 
                     for perf in performance.values()],
        'Kurtosis': [pd.Series(perf['results'].groupby(pd.to_datetime(perf['results']['Date']).dt.to_period('W'))['PnL'].sum()).kurtosis() 
                     for perf in performance.values()],
        'VaR 95%': [perf['metrics']['VaR 95%'] for perf in performance.values()],
        'ES 95%': [perf['metrics']['ES 95%'] for perf in performance.values()],
        'Max Drawdown': [perf['metrics']['Max Drawdown'] for perf in performance.values()],
        'Downside Deviation': [pd.Series(perf['results'].groupby(pd.to_datetime(perf['results']['Date']).dt.to_period('W'))['PnL'].sum())[
                                pd.Series(perf['results'].groupby(pd.to_datetime(perf['results']['Date']).dt.to_period('W'))['PnL'].sum()) < 0].std()
                                for perf in performance.values()]
    })
    
    return {
        'comparison': comparison_table,
        'by_season': season_table,
        'risk_detail': risk_table
    }


def generate_visualizations(results, output_dir='results_output'):
    """Generate all visualizations for the paper"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    performance = results['performance_summary']
    
    # Figure 1: Cumulative P&L Comparison
    plt.figure(figsize=(12, 6))
    for strategy_name, perf_data in performance.items():
        results_df = perf_data['results']
        plt.plot(results_df.index, results_df['CumPnL'], label=strategy_name, linewidth=2)
    
    plt.xlabel('Bet Number', fontsize=12)
    plt.ylabel('Cumulative P&L (£)', fontsize=12)
    plt.title('Cumulative Profit and Loss by Strategy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cumulative_pnl.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Weekly Returns Distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (strategy_name, perf_data) in enumerate(performance.items()):
        results_df = perf_data['results']
        weekly_returns = results_df.groupby(pd.to_datetime(results_df['Date']).dt.to_period('W'))['PnL'].sum()
        
        axes[idx].hist(weekly_returns, bins=30, edgecolor='black', alpha=0.7)
        axes[idx].axvline(weekly_returns.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        axes[idx].axvline(0, color='black', linestyle='-', linewidth=1)
        axes[idx].set_xlabel('Weekly P&L (£)', fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].set_title(strategy_name, fontsize=11, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/weekly_returns_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Risk-Return Scatter
    plt.figure(figsize=(10, 6))
    
    for strategy_name, perf_data in performance.items():
        metrics = perf_data['metrics']
        plt.scatter(metrics['Std Weekly Return'], metrics['Mean Weekly Return'], 
                   s=200, alpha=0.7, label=strategy_name)
    
    plt.xlabel('Standard Deviation of Weekly Returns (£)', fontsize=12)
    plt.ylabel('Mean Weekly Return (£)', fontsize=12)
    plt.title('Risk-Return Trade-off', fontsize=14, fontweight='bold')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/risk_return_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Drawdown Analysis
    plt.figure(figsize=(12, 6))
    
    for strategy_name, perf_data in performance.items():
        results_df = perf_data['results']
        running_max = np.maximum.accumulate(results_df['CumPnL'])
        drawdown = results_df['CumPnL'] - running_max
        plt.plot(results_df.index, drawdown, label=strategy_name, linewidth=2)
    
    plt.xlabel('Bet Number', fontsize=12)
    plt.ylabel('Drawdown (£)', fontsize=12)
    plt.title('Drawdown Analysis', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/drawdown_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Stake Size Distribution (VaR vs Kelly)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    strategies_to_plot = ['VaR Constrained', 'Half Kelly', 'Quarter Kelly']
    for idx, strategy_name in enumerate(strategies_to_plot):
        if strategy_name in performance:
            results_df = performance[strategy_name]['results']
            axes[idx].hist(results_df['Stake'], bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel('Stake Size (£)', fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].set_title(strategy_name, fontsize=11, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/stake_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")


def generate_validation_report(results):
    """Generate model validation statistics"""
    
    validation = results.get('validation_results', {})
    
    report = []
    report.append("="*80)
    report.append("MODEL VALIDATION REPORT")
    report.append("="*80)
    
    for strategy, tests in validation.items():
        report.append(f"\n{strategy}:")
        report.append("-"*80)
        
        if 'kupiec' in tests:
            kupiec = tests['kupiec']
            report.append("\nKupiec Proportion-of-Failures Test:")
            if 'p_value' in kupiec and not np.isnan(kupiec['p_value']):
                report.append(f"  VaR breaches: {kupiec['breaches']}/{kupiec['total_weeks']}")
                report.append(f"  Observed failure rate: {kupiec['failure_rate']:.4f}")
                report.append(f"  Expected failure rate: {kupiec['expected_rate']:.4f}")
                report.append(f"  LR statistic: {kupiec['LR_stat']:.4f}")
                report.append(f"  p-value: {kupiec['p_value']:.4f}")
                report.append(f"  Result: {'REJECT H0 (miscalibrated)' if kupiec['reject_H0'] else 'FAIL TO REJECT H0 (well-calibrated)'}")
            else:
                report.append("  Test not applicable or insufficient data")
        
        if 'independence' in tests:
            indep = tests['independence']
            report.append("\nBet Independence Test:")
            if not np.isnan(indep['mean_correlation']):
                report.append(f"  Weeks tested: {indep['num_weeks_tested']}")
                report.append(f"  Mean correlation: {indep['mean_correlation']:.4f}")
                report.append(f"  Standard error: {indep['se_correlation']:.4f}")
                report.append(f"  t-statistic: {indep['t_stat']:.4f}")
                report.append(f"  p-value: {indep['p_value']:.4f}")
                
                if abs(indep['mean_correlation']) > 0.10:
                    report.append("Material positive correlation")
                elif abs(indep['mean_correlation']) > 0.05:
                    report.append("Moderate correlation")
                else:
                    report.append("Independence assumption reasonable")
            else:
                report.append("  Insufficient data for test")
    
    return "\n".join(report)


def export_to_excel(results, filename='betting_results.xlsx'):
    """Export all results to Excel for easy analysis"""
    
    tables = generate_results_tables(results)
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Main comparison table
        tables['comparison'].to_excel(writer, sheet_name='Strategy Comparison', index=False)
        
        # By-season performance
        tables['by_season'].to_excel(writer, sheet_name='By Season', index=False)
        
        # Risk detail
        tables['risk_detail'].to_excel(writer, sheet_name='Risk Metrics', index=False)
        
        # Detailed bet-level data for each strategy
        for strategy_name, perf_data in results['performance_summary'].items():
            sheet_name = strategy_name[:31]  # Excel sheet name limit
            perf_data['results'].to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Results exported to {filename}")


def generate_latex_tables(results):
    """Generate LaTeX table code for inclusion in paper"""
    
    tables = generate_results_tables(results)
    
    latex_output = []
    
    # Main comparison table
    latex_output.append("% Table: Strategy Performance Comparison")
    latex_output.append(tables['comparison'].to_latex(index=False, float_format="%.2f"))
    latex_output.append("\n")
    
    # Risk metrics table
    latex_output.append("% Table: Risk Metrics Detail")
    latex_output.append(tables['risk_detail'].to_latex(index=False, float_format="%.3f"))
    
    latex_file = 'results_latex_tables.tex'
    with open(latex_file, 'w') as f:
        f.write("\n".join(latex_output))
    
    print(f"LaTeX tables saved to {latex_file}")


def main():
    print("Running full betting analysis...")
    
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'FOOTBALLDATA2')
    
    results = run_analysis(desktop_path)
    
    print("\n" + "="*80)
    print("GENERATING RESULTS FOR SECTION 4")
    print("="*80)
    
    print("\n[1/4] Generating summary tables...")
    tables = generate_results_tables(results)
    print(" Comparison table generated")
    print(" By-season table generated")
    print(" Risk detail table generated")
    
    print("\n[2/4] Generating visualizations...")
    generate_visualizations(results)
    print(" All figures saved to results_output/")
    
    print("\n[3/4] Generating validation report...")
    validation_report = generate_validation_report(results)
    print(validation_report)
    
    with open('validation_report.txt', 'w', encoding='utf-8') as f:
        f.write(validation_report)
    print("\n Validation report saved to validation_report.txt")
    
    print("\n[4/4] Exporting results...")
    export_to_excel(results)
    generate_latex_tables(results)
    
    print("\n" + "="*80)
    print("RESULTS GENERATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - betting_results.xlsx: All results in Excel format")
    print("  - results_latex_tables.tex: Tables formatted for LaTeX")
    print("  - validation_report.txt: Model validation statistics")
    print("  - results_output/: All figures (PNG format)")
    
    return results, tables


if __name__ == "__main__":
    results, tables = main()
