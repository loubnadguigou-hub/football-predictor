import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

class EloDixonColesModel:
    def __init__(self, k_factor: float = 40.0, default_elo: float = 1500.0):
        self.k_factor = k_factor
        self.default_elo = default_elo
        self.elos = {}

        # Default paper parameters
        self.params = {
            'base': 1.11,   # Expected goals for an average neutral match
            'beta': 0.75,   # Sensitivity multiplier for Elo gap
            'hfa': 150.0,   # Home Advantage in Elo points
            'rho': -0.05    # Low-score draw adjustment
        }

    def get_elo(self, team: str) -> float:
        return self.elos.get(team, self.default_elo)

    def get_teams(self) -> list:
        """
        Returns the list of teams the model knows about.
        Falls back to a default EPL team list if no match data
        has been fit yet (self.elos will be empty in that case).
        """
        if self.elos:
            return sorted(self.elos.keys())

        return sorted([
            "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
            "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
            "Leicester", "Liverpool", "Man City", "Man United", "Newcastle",
            "Nottingham Forest", "Southampton", "Tottenham", "West Ham", "Wolves"
        ])

    def _expected_score(self, elo_a: float, elo_b: float, is_home: bool = True) -> float:
        h_boost = self.params['hfa'] if is_home else 0.0
        return 1.0 / (1.0 + 10.0 ** (-(elo_a + h_boost - elo_b) / 400.0))

    def fit_elo_ratings(self, df_matches: pd.DataFrame):
        """
        Iterates over matches sequentially to build dynamic team Elo ratings.
        """
        self.elos = {}
        for _, row in df_matches.iterrows():
            h_team, a_team = row['home_team'], row['away_team']
            h_goals, a_goals = row['home_goals'], row['away_goals']
            is_home = row.get('is_home', True)

            r_h = self.get_elo(h_team)
            r_a = self.get_elo(a_team)

            if h_goals > a_goals:
                w_h, w_a = 1.0, 0.0
            elif h_goals == a_goals:
                w_h, w_a = 0.5, 0.5
            else:
                w_h, w_a = 0.0, 1.0

            e_h = self._expected_score(r_h, r_a, is_home=is_home)
            e_a = 1.0 - e_h

            gd = abs(h_goals - a_goals)
            g_mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11.0 + gd) / 8.0)

            self.elos[h_team] = r_h + self.k_factor * g_mult * (w_h - e_h)
            self.elos[a_team] = r_a + self.k_factor * g_mult * (w_a - e_a)

    @staticmethod
    def tau_correction(x: int, y: int, lambda_a: float, lambda_b: float, rho: float) -> float:
        """
        Dixon-Coles adjustment factor for low scorelines (0-0, 1-0, 0-1, 1-1).
        """
        if x == 0 and y == 0:
            return 1.0 - lambda_a * lambda_b * rho
        elif x == 0 and y == 1:
            return 1.0 + lambda_a * rho
        elif x == 1 and y == 0:
            return 1.0 + lambda_b * rho
        elif x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    def fit_mle_parameters(self, df_matches: pd.DataFrame):
        """
        Optimizes model hyperparameters via Maximum Likelihood Estimation.
        """
        def negative_log_likelihood(params):
            base, beta, hfa, rho = params
            if base <= 0 or beta <= 0 or hfa < 0 or abs(rho) >= 1.0:
                return 1e10

            log_lh = 0.0
            for _, row in df_matches.iterrows():
                r_a = self.get_elo(row['home_team'])
                r_b = self.get_elo(row['away_team'])
                is_home = row.get('is_home', True)

                delta = (r_a - r_b) + (hfa if is_home else 0.0)
                l_a = base * np.exp(beta * (delta / 400.0))
                l_b = base * np.exp(-beta * (delta / 400.0))

                x, y = int(row['home_goals']), int(row['away_goals'])
                p_x = poisson.pmf(x, l_a)
                p_y = poisson.pmf(y, l_b)
                tau = self.tau_correction(x, y, l_a, l_b, rho)

                prob = max(1e-12, p_x * p_y * tau)
                log_lh += np.log(prob)

            return -log_lh

        init_params = [self.params['base'], self.params['beta'], self.params['hfa'], self.params['rho']]
        bounds = [(0.1, 3.0), (0.01, 2.0), (0.0, 300.0), (-0.3, 0.3)]

        res = minimize(negative_log_likelihood, init_params, bounds=bounds, method='L-BFGS-B')
        if res.success:
            self.params['base'], self.params['beta'], self.params['hfa'], self.params['rho'] = res.x

    def predict_match(self, team_a: str, team_b: str, is_home: bool = True, max_goals: int = 6):
        """
        Calculates expected goals, 1X2 outcome probabilities, and a scoreline probability matrix.
        """
        elo_a = self.get_elo(team_a)
        elo_b = self.get_elo(team_b)

        delta = (elo_a - elo_b) + (self.params['hfa'] if is_home else 0.0)
        lambda_a = self.params['base'] * np.exp(self.params['beta'] * (delta / 400.0))
        lambda_b = self.params['base'] * np.exp(-self.params['beta'] * (delta / 400.0))

        goals = np.arange(0, max_goals + 1)
        prob_a = poisson.pmf(goals, lambda_a)
        prob_b = poisson.pmf(goals, lambda_b)
        matrix = np.outer(prob_a, prob_b)

        tau = np.ones_like(matrix)
        tau[0, 0] = 1.0 - lambda_a * lambda_b * self.params['rho']
        tau[0, 1] = 1.0 + lambda_a * self.params['rho']
        tau[1, 0] = 1.0 + lambda_b * self.params['rho']
        tau[1, 1] = 1.0 - self.params['rho']

        adjusted_matrix = np.maximum(0, matrix * tau)
        adjusted_matrix /= np.sum(adjusted_matrix)

        win_a = float(np.tril(adjusted_matrix, -1).sum())
        draw = float(np.trace(adjusted_matrix))
        win_b = float(np.triu(adjusted_matrix, 1).sum())

        return {
            "elo_a": elo_a, "elo_b": elo_b,
            "lambda_a": lambda_a, "lambda_b": lambda_b,
            "win_a": win_a, "draw": draw, "win_b": win_b,
            "score_matrix": adjusted_matrix
        }

    def predict(self, home_team: str, away_team: str, max_goals: int = 6) -> dict:
        """
        Wrapper around predict_match() that returns the key names app.py expects,
        plus the single most likely exact scoreline.
        """
        result = self.predict_match(home_team, away_team, is_home=True, max_goals=max_goals)
        matrix = result["score_matrix"]

        best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        best_home_goals, best_away_goals = int(best_idx[0]), int(best_idx[1])
        best_prob = float(matrix[best_idx])

        return {
            "home_elo": result["elo_a"],
            "away_elo": result["elo_b"],
            "lambda_home": result["lambda_a"],
            "lambda_away": result["lambda_b"],
            "home_win_p": result["win_a"],
            "draw_p": result["draw"],
            "away_win_p": result["win_b"],
            "matrix": matrix,
            "best_home_goals": best_home_goals,
            "best_away_goals": best_away_goals,
            "best_prob": best_prob,
        }
