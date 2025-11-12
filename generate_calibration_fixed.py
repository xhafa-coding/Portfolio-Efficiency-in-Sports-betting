import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from betting_analysis import WalkForwardValidator, SoTPoissonModel, BetSelector
import os

print("="*80)
print("GENERATING CALIBRATION PLOT (WITH PROBABILITY REGENERATION)")
print("="*80)

print("\n[1/3] Regenerating bet data with model probabilities...")

desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'FOOTBALLDATA2')
validator = WalkForwardValidator(desktop_path)
validator.load_data()
splits = validator.get_train_test_splits()

all_bets = []

for split in splits:
    print(f"  Processing {split['test_label']}...")
    
    model = SoTPoissonModel()
    model.fit(split['train'])
    
    selector = BetSelector(model)
    bets_df = selector.generate_bets(split['test'])
    
    if len(bets_df) > 0:
        all_bets.append(bets_df)

# Combine all bets
bet_data = pd.concat(all_bets, ignore_index=True)
print(f"\n  ✓ Total bets with probabilities: {len(bet_data)}")

print("\n[2/3] Creating calibration plot...")

# Extract predicted probability for the outcome that was bet on
def get_predicted_prob(row):
    outcome = row['Outcome']
    if outcome == 'H':
        return row['ProbH']
    elif outcome == 'D':
        return row['ProbD']
    elif outcome == 'A':
        return row['ProbA']
    return np.nan

bet_data['PredictedProb'] = bet_data.apply(get_predicted_prob, axis=1)


bet_data['Win'] = (bet_data['Outcome'] == bet_data['ActualResult']).astype(int)

bet_data = bet_data.dropna(subset=['PredictedProb'])

print(f"  Valid bets for calibration: {len(bet_data)}")

bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bin_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
              '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

bet_data['ProbBin'] = pd.cut(bet_data['PredictedProb'], bins=bins, labels=bin_labels, include_lowest=True)

calibration_data = bet_data.groupby('ProbBin', observed=True).agg({
    'Win': ['mean', 'count'],
    'PredictedProb': 'mean'
}).reset_index()

calibration_data.columns = ['Bin', 'ObservedWinRate', 'Count', 'MeanPredictedProb']


calibration_data = calibration_data[calibration_data['Count'] >= 10]

print(f"\n  Calibration Summary:")
print(calibration_data[['Bin', 'MeanPredictedProb', 'ObservedWinRate', 'Count']].to_string(index=False))

# Calculate overall statistics
overall_predicted = bet_data['PredictedProb'].mean()
overall_observed = bet_data['Win'].mean()
brier_score = np.mean((bet_data['PredictedProb'] - bet_data['Win'])**2)

print(f"\n  Overall Statistics:")
print(f"    Mean Predicted Probability: {overall_predicted:.3f} ({overall_predicted*100:.1f}%)")
print(f"    Actual Win Rate: {overall_observed:.3f} ({overall_observed*100:.1f}%)")
print(f"    Overconfidence: {(overall_predicted - overall_observed):.3f} ({(overall_predicted - overall_observed)*100:.1f} percentage points)")
print(f"    Brier Score: {brier_score:.4f}")


output_dir = 'results_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 3: Generate the plot
print("\n[3/3] Generating calibration plot...")

fig, ax = plt.subplots(figsize=(12, 9))

# Perfect calibration line
ax.plot([0, 1], [0, 1], 'k--', linewidth=2.5, label='Perfect Calibration', alpha=0.7, zorder=1)

# Actual calibration points
scatter = ax.scatter(calibration_data['MeanPredictedProb'], 
                    calibration_data['ObservedWinRate'],
                    s=calibration_data['Count']*3,  # Size proportional to number of bets
                    alpha=0.7, 
                    c='red',
                    edgecolors='darkred',
                    linewidth=2,
                    label='Observed Win Rate',
                    zorder=3)

# Connect the points with a line
if len(calibration_data) > 1:
    ax.plot(calibration_data['MeanPredictedProb'], 
            calibration_data['ObservedWinRate'],
            'r-', 
            linewidth=2.5, 
            alpha=0.6,
            zorder=2)

# Add labels showing counts
for idx, row in calibration_data.iterrows():
   
    if row['ObservedWinRate'] < row['MeanPredictedProb']:
        xytext = (0, -15)
        va = 'top'
    else:
        xytext = (0, 15)
        va = 'bottom'
    
    ax.annotate(f"n={int(row['Count'])}", 
                xy=(row['MeanPredictedProb'], row['ObservedWinRate']),
                xytext=xytext, 
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                ha='center',
                va=va,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.8))

ax.set_xlabel('Model Predicted Win Probability', fontsize=14, fontweight='bold')
ax.set_ylabel('Observed Win Rate', fontsize=14, fontweight='bold')
ax.set_title('Model Calibration: Predicted vs Observed Win Rates', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

if len(calibration_data) > 0:
    x_fill = np.linspace(0, 1, 100)
    y_lower = np.zeros_like(x_fill)
    ax.fill_between(x_fill, y_lower, x_fill, alpha=0.1, color='red', 
                    label='Overconfidence Region', zorder=0)

# Add text box with key findings
overconf_pct = (overall_predicted - overall_observed) * 100
textstr = f'Model Overconfidence:\n{overconf_pct:.1f} percentage points\n\nBrier Score: {brier_score:.3f}\n(0.25 = random)'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1.5)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(f'{output_dir}/model_calibration.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n  ✓ Saved: {output_dir}/model_calibration.png")

# Print summary for paper
print("\n" + "="*80)
print("CALIBRATION RESULTS SUMMARY")
print("="*80)

print(f"\nKey Findings:")
print(f"  1. Model predicted win probability: {overall_predicted*100:.1f}%")
print(f"  2. Actual win rate: {overall_observed*100:.1f}%")
print(f"  3. Overconfidence: {overconf_pct:.1f} percentage points")
print(f"  4. Brier Score: {brier_score:.4f} (vs 0.25 for random guessing)")

if overall_predicted > overall_observed:
    print(f"\n  → Model is OVERCONFIDENT (predicts higher win rates than observed)")
    print(f"  → This explains negative returns across all strategies")
else:
    print(f"\n  → Model is UNDERCONFIDENT (predicts lower win rates than observed)")

print("\n" + "="*80)
print("CHART GENERATED SUCCESSFULLY")
print("="*80)
print(f"\nFor your paper's Results section:")
print(f"  'The calibration analysis reveals systematic model overconfidence,")
print(f"   with predicted win probabilities exceeding observed rates by")
print(f"   {overconf_pct:.1f} percentage points (p={overall_predicted:.3f} vs o={overall_observed:.3f}).")
print(f"   This miscalibration explains the negative returns, as the model")
print(f"   systematically overestimated betting value against efficient")
print(f"   closing odds.'")
print("="*80)
