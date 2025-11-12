import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("="*80)
print("GENERATING ADDITIONAL CHARTS")
print("="*80)

# Load the Excel file
print("\n[1/2] Loading betting results...")
excel_file = 'betting_results.xlsx'

if not os.path.exists(excel_file):
    print(f"ERROR: {excel_file} not found!")
    print("Please run generate_results.py first to create the Excel file.")
    exit(1)

# Load all strategy tabs
strategies = ['Unconstrained', 'VaR Constrained', 'Half Kelly', 'Quarter Kelly']
all_data = {}

for strategy in strategies:
    try:
        df = pd.read_excel(excel_file, sheet_name=strategy)
        all_data[strategy] = df
        print(f"  ✓ Loaded {strategy}: {len(df)} bets")
    except Exception as e:
        print(f"  ✗ Could not load {strategy}: {e}")

output_dir = 'results_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# CHART 1: Model Calibration Plot
print("\n[2/2] Generating Model Calibration Plot...")

# Use Unconstrained strategy data (same bets for all strategies)
bet_data = all_data['Unconstrained'].copy()

# Add predicted probability column (the model's predicted win probability for the bet placed)
# Need to extract the correct probability based on outcome type
def get_predicted_prob(row):
    # The bet was placed on row['Outcome'], so get that probability
    if pd.isna(row['Outcome']):
        return np.nan
    
    outcome = row['Outcome']
    if outcome == 'H' and 'ProbH' in row:
        return row['ProbH']
    elif outcome == 'D' and 'ProbD' in row:
        return row['ProbD']
    elif outcome == 'A' and 'ProbA' in row:
        return row['ProbA']
    return np.nan

bet_data['PredictedProb'] = bet_data.apply(get_predicted_prob, axis=1)

# Remove any NaN probabilities
bet_data = bet_data.dropna(subset=['PredictedProb'])

# Create probability bins
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bin_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
              '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

bet_data['ProbBin'] = pd.cut(bet_data['PredictedProb'], bins=bins, labels=bin_labels, include_lowest=True)

# Calculate observed win rate in each bin
calibration_data = bet_data.groupby('ProbBin', observed=True).agg({
    'Win': ['mean', 'count'],
    'PredictedProb': 'mean'
}).reset_index()

calibration_data.columns = ['Bin', 'ObservedWinRate', 'Count', 'MeanPredictedProb']

# Filter out bins with very few observations
calibration_data = calibration_data[calibration_data['Count'] >= 10]

print(f"  Bins with sufficient data: {len(calibration_data)}")
print("\n  Calibration Summary:")
print(calibration_data[['Bin', 'MeanPredictedProb', 'ObservedWinRate', 'Count']].to_string(index=False))

# Create calibration plot
fig, ax = plt.subplots(figsize=(10, 8))

# Perfect calibration line
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)

# Actual calibration
ax.scatter(calibration_data['MeanPredictedProb'], 
          calibration_data['ObservedWinRate'],
          s=calibration_data['Count']*2,  # Size proportional to number of bets
          alpha=0.6, 
          color='red',
          edgecolors='black',
          linewidth=1.5,
          label='Observed Win Rate')

# Connect the points
ax.plot(calibration_data['MeanPredictedProb'], 
        calibration_data['ObservedWinRate'],
        'r-', 
        linewidth=2, 
        alpha=0.5)

# Add labels showing bin names and counts
for idx, row in calibration_data.iterrows():
    ax.annotate(f"{row['Bin']}\n(n={int(row['Count'])})", 
                xy=(row['MeanPredictedProb'], row['ObservedWinRate']),
                xytext=(10, 10), 
                textcoords='offset points',
                fontsize=8,
                alpha=0.7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax.set_xlabel('Predicted Win Probability', fontsize=14, fontweight='bold')
ax.set_ylabel('Observed Win Rate', fontsize=14, fontweight='bold')
ax.set_title('Model Calibration: Predicted vs Observed Win Rates', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

# Add text box with interpretation
textstr = 'Points below diagonal:\nModel overconfident\n(overestimates win probability)'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(f'{output_dir}/model_calibration.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: {output_dir}/model_calibration.png")

# CHART 2: Performance by Season
print("\n[3/3] Generating Performance by Season...")

# Extract season from date for each strategy
season_performance = []

for strategy_name, df in all_data.items():
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    
    # Map year to season label
    def get_season(year):
        if year == 2019:
            return '2019/20'
        elif year == 2020:
            return '2020/21'
        elif year == 2021:
            return '2021/22'
        elif year == 2022:
            return '2022/23'
        elif year == 2023 or year == 2024:
            return '2023/24'
        return 'Unknown'
    
    df['Season'] = df['Year'].apply(get_season)
    
    # Calculate P&L by season
    season_pnl = df.groupby('Season')['PnL'].sum().reset_index()
    season_pnl['Strategy'] = strategy_name
    season_performance.append(season_pnl)

season_df = pd.concat(season_performance, ignore_index=True)

print("\n  Season Performance Summary:")
print(season_df.pivot(index='Season', columns='Strategy', values='PnL').to_string())

# Create grouped bar chart
seasons_ordered = ['2019/20', '2020/21', '2021/22', '2022/23', '2023/24']
season_df = season_df[season_df['Season'].isin(seasons_ordered)]

fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(seasons_ordered))
width = 0.2

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, strategy in enumerate(strategies):
    strategy_data = season_df[season_df['Strategy'] == strategy]
    strategy_data = strategy_data.set_index('Season').reindex(seasons_ordered, fill_value=0)
    pnl_values = strategy_data['PnL'].values
    
    ax.bar(x + i*width, pnl_values, width, 
           label=strategy, 
           color=colors[i],
           edgecolor='black',
           linewidth=1)
    
    # Add value labels on bars
    for j, v in enumerate(pnl_values):
        if abs(v) > 50:  # Only label if P&L is significant
            ax.text(j + i*width, v, f'£{int(v)}', 
                   ha='center', 
                   va='bottom' if v >= 0 else 'top',
                   fontsize=8,
                   fontweight='bold')

ax.set_xlabel('Season', fontsize=14, fontweight='bold')
ax.set_ylabel('Profit & Loss (£)', fontsize=14, fontweight='bold')
ax.set_title('Strategy Performance by Season', fontsize=16, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(seasons_ordered, fontsize=12)
ax.legend(fontsize=11, loc='upper right')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_dir}/performance_by_season.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: {output_dir}/performance_by_season.png")

# Summary Statistics
print("\n" + "="*80)
print("CALIBRATION STATISTICS")
print("="*80)

# Calculate overall calibration metrics
bet_data_full = all_data['Unconstrained'].copy()
bet_data_full['PredictedProb'] = bet_data_full.apply(get_predicted_prob, axis=1)
bet_data_full = bet_data_full.dropna(subset=['PredictedProb'])

overall_predicted = bet_data_full['PredictedProb'].mean()
overall_observed = bet_data_full['Win'].mean()

print(f"\nOverall Model Performance:")
print(f"  Mean Predicted Win Probability: {overall_predicted:.3f} ({overall_predicted*100:.1f}%)")
print(f"  Actual Win Rate: {overall_observed:.3f} ({overall_observed*100:.1f}%)")
print(f"  Overconfidence: {overall_predicted - overall_observed:.3f} ({(overall_predicted - overall_observed)*100:.1f}%)")

# Brier Score
brier_score = np.mean((bet_data_full['PredictedProb'] - bet_data_full['Win'])**2)
print(f"\nBrier Score: {brier_score:.4f}")
print(f"  (Lower is better; 0.25 = random guessing)")

print("\n" + "="*80)
print("CHARTS GENERATED SUCCESSFULLY")
print("="*80)
print(f"\nNew charts saved in {output_dir}/:")
print("  1. model_calibration.png")
print("  2. performance_by_season.png")
print("\nThese charts are critical for explaining:")
print("  - WHY the model failed (calibration)")
print("  - CONSISTENCY of results across seasons")
print("="*80)
