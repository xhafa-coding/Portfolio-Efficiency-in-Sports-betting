import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.special import factorial
import warnings
warnings.filterwarnings('ignore')

# 3.1.1 Walk-Forward Validation Framework

class WalkForwardValidator:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.seasons = ['1819', '1920', '2021', '2122', '2223', '2324']
        self.season_labels = ['2018/19', '2019/20', '2020/21', '2021/22', '2022/23', '2023/24']
        self.data = {}
        
    def load_data(self):
        for season in self.seasons:
            file_path = self.data_path / f'{season}.csv'
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            df = df.sort_values('Date').reset_index(drop=True)
            self.data[season] = df
        return self.data
    
    def get_train_test_splits(self):
        splits = []
        for i in range(1, len(self.seasons)):
            train_season = self.seasons[i-1]
            test_season = self.seasons[i]
            splits.append({
                'train': self.data[train_season],
                'test': self.data[test_season],
                'train_label': self.season_labels[i-1],
                'test_label': self.season_labels[i]
            })
        return splits


# 3.2.1 SoT Poisson Specification

class SoTPoissonModel:
    def __init__(self):
        self.params = {}
        
    def fit(self, train_data):
        df = train_data.copy()
        
        self.params['mu_HST'] = df['HST'].mean()
        self.params['mu_AST'] = df['AST'].mean()
        
        total_goals = df['FTHG'].sum() + df['FTAG'].sum()
        total_sot = df['HST'].sum() + df['AST'].sum()
        self.params['conversion_rate'] = total_goals / total_sot
        
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        self.params['teams'] = teams
        
        alpha_H = {}
        delta_H = {}
        alpha_A = {}
        delta_A = {}
        
        for team in teams:
            home_matches = df[df['HomeTeam'] == team]
            away_matches = df[df['AwayTeam'] == team]
            
            if len(home_matches) > 0:
                alpha_H[team] = home_matches['HST'].mean() / self.params['mu_HST']
                delta_H[team] = home_matches['AST'].mean() / self.params['mu_AST']
            else:
                alpha_H[team] = 1.0
                delta_H[team] = 1.0
            
            if len(away_matches) > 0:
                alpha_A[team] = away_matches['AST'].mean() / self.params['mu_AST']
                delta_A[team] = away_matches['HST'].mean() / self.params['mu_HST']
            else:
                alpha_A[team] = 1.0
                delta_A[team] = 1.0
        
        self.params['alpha_H'] = alpha_H
        self.params['delta_H'] = delta_H
        self.params['alpha_A'] = alpha_A
        self.params['delta_A'] = delta_A
        
        return self
    
    def predict_match(self, home_team, away_team):
        if home_team not in self.params['teams'] or away_team not in self.params['teams']:
            return None
        
        mu_H = (self.params['alpha_H'][home_team] * 
                self.params['delta_A'][away_team] * 
                self.params['mu_HST'])
        
        mu_A = (self.params['alpha_A'][away_team] * 
                self.params['delta_H'][home_team] * 
                self.params['mu_AST'])
        
        lambda_H = mu_H * self.params['conversion_rate']
        lambda_A = mu_A * self.params['conversion_rate']
        
        return lambda_H, lambda_A
    
    def calculate_match_probabilities(self, lambda_H, lambda_A, max_goals=10):
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob_matrix[h, a] = (
                    (lambda_H**h * np.exp(-lambda_H) / factorial(h)) *
                    (lambda_A**a * np.exp(-lambda_A) / factorial(a))
                )
        
        prob_home = np.sum(np.tril(prob_matrix, -1))
        prob_draw = np.sum(np.diag(prob_matrix))
        prob_away = np.sum(np.triu(prob_matrix, 1))
        
        total = prob_home + prob_draw + prob_away
        
        return {
            'H': prob_home / total,
            'D': prob_draw / total,
            'A': prob_away / total
        }


# 3.2.3 Probability Generation and Bet Selection

class BetSelector:
    def __init__(self, model):
        self.model = model
        
    def generate_bets(self, test_data):
        bets = []
        
        for idx, row in test_data.iterrows():
            prediction = self.model.predict_match(row['HomeTeam'], row['AwayTeam'])
            
            if prediction is None:
                continue
            
            lambda_H, lambda_A = prediction
            probs = self.model.calculate_match_probabilities(lambda_H, lambda_A)
            
            outcomes = {
                'H': ('B365CH', probs['H']),
                'D': ('B365CD', probs['D']),
                'A': ('B365CA', probs['A'])
            }
            
            for outcome, (odds_col, model_prob) in outcomes.items():
                if odds_col not in row or pd.isna(row[odds_col]):
                    continue
                    
                market_odds = row[odds_col]
                ev = model_prob * market_odds - 1
                
                if ev > 0:
                    bets.append({
                        'Date': row['Date'],
                        'HomeTeam': row['HomeTeam'],
                        'AwayTeam': row['AwayTeam'],
                        'Outcome': outcome,
                        'ModelProb': model_prob,
                        'MarketOdds': market_odds,
                        'EV': ev,
                        'ActualResult': row['FTR'],
                        'ProbH': probs['H'],
                        'ProbD': probs['D'],
                        'ProbA': probs['A']
                    })
        
        return pd.DataFrame(bets)


# 3.3.2 Unconstrained Fixed Staking

class UnconstrainedStrategy:
    def __init__(self, fixed_stake=10):
        self.fixed_stake = fixed_stake
        self.results = []
        
    def execute(self, bets_df):
        results = []
        
        for idx, bet in bets_df.iterrows():
            stake = self.fixed_stake
            win = bet['Outcome'] == bet['ActualResult']
            pnl = stake * (bet['MarketOdds'] - 1) if win else -stake
            
            results.append({
                'Date': bet['Date'],
                'Stake': stake,
                'Outcome': bet['Outcome'],
                'Win': win,
                'PnL': pnl,
                'CumPnL': 0
            })
        
        results_df = pd.DataFrame(results)
        results_df['CumPnL'] = results_df['PnL'].cumsum()
        self.results = results_df
        
        return results_df


# 3.3.3 Value at Risk Constrained Staking

class VaRConstrainedStrategy:
    def __init__(self, fixed_stake=10, var_limit=-50, n_simulations=5000):
        self.fixed_stake = fixed_stake
        self.var_limit = var_limit
        self.n_simulations = n_simulations
        self.results = []
        
    def simulate_weekly_pnl(self, weekly_bets):
        simulations = []
        
        for _ in range(self.n_simulations):
            weekly_pnl = 0
            
            for _, bet in weekly_bets.iterrows():
                probs = [bet['ProbH'], bet['ProbD'], bet['ProbA']]
                outcome = np.random.choice(['H', 'D', 'A'], p=probs)
                
                if outcome == bet['Outcome']:
                    weekly_pnl += self.fixed_stake * (bet['MarketOdds'] - 1)
                else:
                    weekly_pnl -= self.fixed_stake
            
            simulations.append(weekly_pnl)
        
        return np.array(simulations)
    
    def calculate_var(self, simulations):
        return np.percentile(simulations, 5)
    
    def execute(self, bets_df):
        bets_df['Week'] = bets_df['Date'].dt.to_period('W')
        weeks = bets_df['Week'].unique()
        results = []
        
        for week in weeks:
            weekly_bets = bets_df[bets_df['Week'] == week]
            
            if len(weekly_bets) == 0:
                continue
            
            simulations = self.simulate_weekly_pnl(weekly_bets)
            var_estimate = self.calculate_var(simulations)
            
            if var_estimate >= self.var_limit:
                scaling_factor = 1.0
            else:
                scaling_factor = self.var_limit / var_estimate
            
            for idx, bet in weekly_bets.iterrows():
                stake = self.fixed_stake * scaling_factor
                win = bet['Outcome'] == bet['ActualResult']
                pnl = stake * (bet['MarketOdds'] - 1) if win else -stake
                
                results.append({
                    'Date': bet['Date'],
                    'Stake': stake,
                    'ScalingFactor': scaling_factor,
                    'WeeklyVaR': var_estimate,
                    'Outcome': bet['Outcome'],
                    'Win': win,
                    'PnL': pnl,
                    'CumPnL': 0
                })
        
        results_df = pd.DataFrame(results)
        results_df['CumPnL'] = results_df['PnL'].cumsum()
        self.results = results_df
        
        return results_df


# 3.3.4 Kelly Criterion

class KellyStrategy:
    def __init__(self, initial_bankroll=1000, fraction=0.5):
        self.initial_bankroll = initial_bankroll
        self.fraction = fraction
        self.bankroll = initial_bankroll
        self.results = []
        
    def calculate_kelly_stake(self, prob, odds):
        kelly_fraction = (prob * odds - 1) / (odds - 1)
        kelly_stake = self.fraction * kelly_fraction * self.bankroll
        return max(0, kelly_stake)
    
    def execute(self, bets_df):
        results = []
        self.bankroll = self.initial_bankroll
        
        for idx, bet in bets_df.iterrows():
            stake = self.calculate_kelly_stake(bet['ModelProb'], bet['MarketOdds'])
            stake = min(stake, self.bankroll * 0.25)
            
            if stake < 0.01:
                continue
            
            win = bet['Outcome'] == bet['ActualResult']
            pnl = stake * (bet['MarketOdds'] - 1) if win else -stake
            
            self.bankroll += pnl
            
            results.append({
                'Date': bet['Date'],
                'Stake': stake,
                'Bankroll': self.bankroll,
                'Outcome': bet['Outcome'],
                'Win': win,
                'PnL': pnl,
                'CumPnL': 0
            })
        
        results_df = pd.DataFrame(results)
        results_df['CumPnL'] = results_df['PnL'].cumsum()
        self.results = results_df
        
        return results_df


# 3.4 Risk Estimation

class RiskEstimator:
    @staticmethod
    def calculate_historical_var(returns, confidence=0.95):
        if len(returns) == 0:
            return np.nan
        sorted_returns = np.sort(returns)
        index = int((1 - confidence) * len(returns))
        return sorted_returns[index] if index < len(returns) else sorted_returns[0]
    
    @staticmethod
    def calculate_historical_es(returns, confidence=0.95):
        if len(returns) == 0:
            return np.nan
        var = RiskEstimator.calculate_historical_var(returns, confidence)
        tail_returns = returns[returns <= var]
        return tail_returns.mean() if len(tail_returns) > 0 else np.nan
    
    @staticmethod
    def calculate_maximum_drawdown(cumulative_returns):
        if len(cumulative_returns) == 0:
            return 0
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        return drawdown.min()


# 3.5 Performance Evaluation

class PerformanceEvaluator:
    def __init__(self, results_df):
        self.results = results_df
        
    def calculate_metrics(self):
        if len(self.results) == 0:
            return {}
        
        total_bets = len(self.results)
        total_pnl = self.results['PnL'].sum()
        wins = self.results['Win'].sum()
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        avg_stake = self.results['Stake'].mean()
        total_staked = self.results['Stake'].sum()
        roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0
        
        self.results['Week'] = pd.to_datetime(self.results['Date']).dt.to_period('W')
        weekly_returns = self.results.groupby('Week')['PnL'].sum()
        
        mean_weekly_return = weekly_returns.mean()
        std_weekly_return = weekly_returns.std()
        
        var_95 = RiskEstimator.calculate_historical_var(weekly_returns.values, 0.95)
        es_95 = RiskEstimator.calculate_historical_es(weekly_returns.values, 0.95)
        
        mdd = RiskEstimator.calculate_maximum_drawdown(self.results['CumPnL'].values)
        
        sharpe = mean_weekly_return / std_weekly_return if std_weekly_return > 0 else 0
        
        downside_returns = weekly_returns[weekly_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else std_weekly_return
        sortino = mean_weekly_return / downside_std if downside_std > 0 else 0
        
        calmar = total_pnl / abs(mdd) if mdd != 0 else 0
        
        positive_returns = weekly_returns[weekly_returns > 0].sum()
        negative_returns = abs(weekly_returns[weekly_returns < 0].sum())
        omega = positive_returns / negative_returns if negative_returns > 0 else np.inf
        
        metrics = {
            'Total Bets': total_bets,
            'Total P&L': total_pnl,
            'Win Rate': win_rate,
            'Avg Stake': avg_stake,
            'Total Staked': total_staked,
            'ROI (%)': roi,
            'Mean Weekly Return': mean_weekly_return,
            'Std Weekly Return': std_weekly_return,
            'VaR 95%': var_95,
            'ES 95%': es_95,
            'Max Drawdown': mdd,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Calmar Ratio': calmar,
            'Omega Ratio': omega,
            'Num Weeks': len(weekly_returns)
        }
        
        return metrics


# 3.6 Model Validation

class ModelValidator:
    @staticmethod
    def kupiec_pof_test(results_df, confidence=0.95):
        results_df['Week'] = pd.to_datetime(results_df['Date']).dt.to_period('W')
        weekly_returns = results_df.groupby('Week')['PnL'].sum()
        
        if 'WeeklyVaR' not in results_df.columns:
            return {'test': 'N/A', 'p_value': np.nan}
        
        weeks_with_var = results_df.groupby('Week')['WeeklyVaR'].first()
        
        breaches = sum(weekly_returns.values <= weeks_with_var.values)
        n_weeks = len(weekly_returns)
        
        p_star = 1 - confidence
        p_hat = breaches / n_weeks
        
        if p_hat == 0 or p_hat == 1:
            return {'breaches': breaches, 'total_weeks': n_weeks, 'failure_rate': p_hat, 
                    'LR_stat': np.nan, 'p_value': np.nan}
        
        lr_stat = -2 * (
            (n_weeks - breaches) * np.log((1 - p_star) / (1 - p_hat)) +
            breaches * np.log(p_star / p_hat)
        )
        
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return {
            'breaches': breaches,
            'total_weeks': n_weeks,
            'failure_rate': p_hat,
            'expected_rate': p_star,
            'LR_stat': lr_stat,
            'p_value': p_value,
            'reject_H0': p_value < 0.05
        }
    
    @staticmethod
    def test_bet_independence(results_df):
        results_df['Week'] = pd.to_datetime(results_df['Date']).dt.to_period('W')
        weeks = results_df['Week'].unique()
        
        weekly_correlations = []
        
        for week in weeks:
            week_bets = results_df[results_df['Week'] == week]
            if len(week_bets) < 2:
                continue
            
            wins = week_bets['Win'].astype(int).values
            if len(np.unique(wins)) < 2:
                continue
            
            corr_matrix = np.corrcoef(wins[:-1], wins[1:])
            if not np.isnan(corr_matrix[0, 1]):
                weekly_correlations.append(corr_matrix[0, 1])
        
        if len(weekly_correlations) == 0:
            return {'mean_correlation': np.nan, 't_stat': np.nan, 'p_value': np.nan}
        
        mean_corr = np.mean(weekly_correlations)
        se_corr = np.std(weekly_correlations) / np.sqrt(len(weekly_correlations))
        t_stat = mean_corr / se_corr if se_corr > 0 else 0
        p_value = 1 - stats.t.cdf(t_stat, df=len(weekly_correlations)-1)
        
        return {
            'num_weeks_tested': len(weekly_correlations),
            'mean_correlation': mean_corr,
            'se_correlation': se_corr,
            't_stat': t_stat,
            'p_value': p_value
        }


# Main execution pipeline

def run_analysis(data_path):
    print("="*80)
    print("SPORTS BETTING PORTFOLIO ANALYSIS")
    print("VaR Constraints vs Kelly Criterion")
    print("="*80)
    
    validator = WalkForwardValidator(data_path)
    
    print("\n[1/6] Loading data...")
    validator.load_data()
    splits = validator.get_train_test_splits()
    print(f"Loaded {len(splits)} train-test splits")
    
    all_results = {
        'Unconstrained': [],
        'VaR Constrained': [],
        'Half Kelly': [],
        'Quarter Kelly': []
    }
    
    all_bets_by_season = []
    
    for idx, split in enumerate(splits):
        print(f"\n[2/6] Processing {split['test_label']}...")
        print(f"  Training on: {split['train_label']}")
        print(f"  Testing on: {split['test_label']}")
        
        model = SoTPoissonModel()
        model.fit(split['train'])
        
        print(f"  Model parameters estimated:")
        print(f"    μ_HST = {model.params['mu_HST']:.2f}")
        print(f"    μ_AST = {model.params['mu_AST']:.2f}")
        print(f"    Conversion rate = {model.params['conversion_rate']:.3f}")
        
        selector = BetSelector(model)
        bets_df = selector.generate_bets(split['test'])
        
        print(f"  Positive EV bets identified: {len(bets_df)}")
        
        if len(bets_df) > 0:
            all_bets_by_season.append({
                'season': split['test_label'],
                'bets': bets_df
            })
        
        if len(bets_df) == 0:
            print("  No positive EV bets found, skipping...")
            continue
        
        print(f"\n[3/6] Executing strategies for {split['test_label']}...")
        
        unconstrained = UnconstrainedStrategy(fixed_stake=10)
        unconstrained_results = unconstrained.execute(bets_df)
        all_results['Unconstrained'].append(unconstrained_results)
        
        var_constrained = VaRConstrainedStrategy(fixed_stake=10, var_limit=-50, n_simulations=5000)
        var_results = var_constrained.execute(bets_df)
        all_results['VaR Constrained'].append(var_results)
        
        half_kelly = KellyStrategy(initial_bankroll=1000, fraction=0.5)
        half_kelly_results = half_kelly.execute(bets_df)
        all_results['Half Kelly'].append(half_kelly_results)
        
        quarter_kelly = KellyStrategy(initial_bankroll=1000, fraction=0.25)
        quarter_kelly_results = quarter_kelly.execute(bets_df)
        all_results['Quarter Kelly'].append(quarter_kelly_results)
        
        print(f"  ✓ All strategies executed")
    
    print("\n[4/6] Calculating performance metrics...")
    
    performance_summary = {}
    
    for strategy_name, results_list in all_results.items():
        if len(results_list) == 0:
            continue
        
        combined_results = pd.concat(results_list, ignore_index=True)
        combined_results['CumPnL'] = combined_results['PnL'].cumsum()
        
        evaluator = PerformanceEvaluator(combined_results)
        metrics = evaluator.calculate_metrics()
        
        performance_summary[strategy_name] = {
            'metrics': metrics,
            'results': combined_results
        }
        
        print(f"\n  {strategy_name}:")
        print(f"    Total P&L: £{metrics['Total P&L']:.2f}")
        print(f"    ROI: {metrics['ROI (%)']:.2f}%")
        print(f"    Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
        print(f"    Max Drawdown: £{metrics['Max Drawdown']:.2f}")
    
    print("\n[5/6] Running model validation tests...")
    
    validation_results = {}
    
    for strategy_name in ['VaR Constrained']:
        if strategy_name in performance_summary:
            results_df = performance_summary[strategy_name]['results']
            
            kupiec = ModelValidator.kupiec_pof_test(results_df)
            independence = ModelValidator.test_bet_independence(results_df)
            
            validation_results[strategy_name] = {
                'kupiec': kupiec,
                'independence': independence
            }
            
            print(f"\n  {strategy_name} - Kupiec POF Test:")
            if 'p_value' in kupiec and not np.isnan(kupiec['p_value']):
                print(f"    Breaches: {kupiec['breaches']}/{kupiec['total_weeks']}")
                print(f"    p-value: {kupiec['p_value']:.4f}")
                print(f"    Result: {'REJECT H0' if kupiec['reject_H0'] else 'FAIL TO REJECT H0'}")
    
    print("\n[6/6] Generating summary statistics...")
    
    summary_df = pd.DataFrame({
        strategy: {
            'Total P&L (£)': perf['metrics']['Total P&L'],
            'Total Bets': perf['metrics']['Total Bets'],
            'Win Rate (%)': perf['metrics']['Win Rate'] * 100,
            'ROI (%)': perf['metrics']['ROI (%)'],
            'Sharpe Ratio': perf['metrics']['Sharpe Ratio'],
            'Sortino Ratio': perf['metrics']['Sortino Ratio'],
            'Calmar Ratio': perf['metrics']['Calmar Ratio'],
            'Omega Ratio': perf['metrics']['Omega Ratio'],
            'Max Drawdown (£)': perf['metrics']['Max Drawdown'],
            'VaR 95% (£)': perf['metrics']['VaR 95%'],
            'ES 95% (£)': perf['metrics']['ES 95%']
        }
        for strategy, perf in performance_summary.items()
    }).T
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.to_string())
    
    return {
        'performance_summary': performance_summary,
        'validation_results': validation_results,
        'summary_df': summary_df,
        'all_bets_by_season': all_bets_by_season
    }


if __name__ == "__main__":
    import os
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'FOOTBALLDATA2')
    
    results = run_analysis(desktop_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nResults saved in 'results' dictionary with keys:")
    print("  - performance_summary: detailed metrics for each strategy")
    print("  - validation_results: Kupiec and independence tests")
    print("  - summary_df: comparison table of all strategies")
    print("  - all_bets_by_season: bet-level data by season")
