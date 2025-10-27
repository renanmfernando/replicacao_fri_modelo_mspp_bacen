import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve, inv
from scipy.optimize import fsolve
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelParameters:
    """Class to store all model parameters"""
    # IS Curve parameters
    beta_1: float = 0.85
    beta_2: float = 0.44
    beta_3: float = 0.030
    beta_4: float = 0.054
    beta_5: float = 0.84
    
    # Phillips Curve parameters
    alp_1_L: float = 0.24
    alp_1_I: float = 0.38
    alp_2: float = 0.023
    alp_3: float = 0.011
    alp_4: float = 0.120
    
    # Weights
    w_icbr_a: float = 0.64
    w_icbr_m: float = 0.19
    w_adm: float = 0.2509
    
    # Taylor Rule parameters
    theta_1: float = 1.48
    theta_2: float = -0.58
    theta_3: float = 2.03
    
    # UIP parameters
    delta: float = 1.90
    
    # Expectations parameters
    phi_1: float = 0.75
    phi_2: float = 0.11
    phi_3: float = 0.021
    
    # Administered prices parameters
    gama_1: float = 0.5
    gama_2: float = 0.12
    gama_3: float = 0.012
    gama_4: float = 0.012
    
    # Steady state values
    meta_ss: float = 3.0
    infl_ss_ext: float = 2.0

class BacenMSPPModel:
    def __init__(self, params: ModelParameters = None):
        self.params = params if params else ModelParameters()
        
        # Define variable names (maintaining exact order from .mod file)
        self.endogenous_vars = [
            'hiato_prod',           # 0
            'juros_real_gap',       # 1
            'exp_selic_focus_12m',  # 2
            'exp_infl_focus_12m',   # 3
            'juros_real_neutro',    # 4
            'selic',                # 5
            'juros_neutro_gap',     # 6
            'shk_hiato_prod',       # 7
            'infl_cheia_tri',       # 8
            'infl_livres_tri',      # 9
            'infl_livres_12m',      # 10
            'infl_admin_12m',       # 11
            'infl_cheia_12m',       # 12
            'desv_inf_importada',   # 13
            'desv_cambio',          # 14
            'desv_icbr_a',          # 15
            'desv_icbr_m',          # 16
            'desv_icbr_e',          # 17
            'icbr_a',               # 18
            'icbr_m',               # 19
            'icbr_e',               # 20
            'meta_infl',            # 21
            'var_cambio',           # 22
            'juros_neutro_taylor',  # 23
            'juros_neutro_taylor_gap', # 24
            'var_ppc',              # 25
            'dif_juros',            # 26
            'exp_infl_consistente_modelo', # 27
            'brent',                # 28
            'infl_admin_tri',       # 29
            'var_cambio_acum'       # 30
        ]
        
        self.exogenous_vars = [
            'e_selic_is',           # 0
            'e_juros_neutro_gap',   # 1
            'primario_governo',     # 2
            'hiato_mundial',        # 3
            'e_shk_hiato_prod',     # 4
            'e_phillips',           # 5
            'e_icbr_a',             # 6
            'e_icbr_m',             # 7
            'e_icbr_e',             # 8
            'e_meta',               # 9
            'e_taylor',             # 10
            'e_juros_neutro_taylor_gap', # 11
            'e_uip',                # 12
            'fed_funds',            # 13
            'cds',                  # 14
            'e_exp',                # 15
            'e_hiato_prod',         # 16
            'juros_neutro_trend',   # 17
            'e_brent'               # 18
        ]
        
        # Shock standard deviations
        self.shock_stds = {
            'e_uip': 10,
            'e_hiato_prod': 1,
            'e_taylor': 3,
            'e_juros_neutro_taylor_gap': 1,
            'e_exp': 1,
            'e_brent': 10
        }
        
        self.n_endo = len(self.endogenous_vars)
        self.n_exo = len(self.exogenous_vars)
        
        # Create state space representation
        self.setup_state_space()
    
    def setup_state_space(self):
        """Setup state space representation for the linear MSPP model"""
        # For a linear MSPP model: E[y(t+1)] = A * y(t) + B * x(t)
        # We need to solve this properly considering leads and lags
        
        # This is a simplified approach - in practice, we'd need to solve
        # the full rational expectations system using methods like Blanchard-Kahn
        self.n_states = self.n_endo + 4  # Add lags for selic(-2), and other multi-lag variables
        
        # We'll store the last few periods for proper lag handling
        self.max_lags = 4
        
    def get_var_value(self, y_full, var_name, lag=0):
        """Get variable value with proper lag handling"""
        var_idx = self.endogenous_vars.index(var_name)
        period_idx = self.max_lags + lag
        if period_idx < 0 or period_idx >= y_full.shape[0]:
            return 0.0
        return y_full[period_idx, var_idx]
    
    def model_equations_full(self, y_full, x_t, period_t):
        """
        Complete model equations with proper forward/backward looking terms
        y_full: array with multiple time periods [t-max_lags, ..., t, ..., t+max_lags]
        x_t: exogenous variables at time t
        period_t: index of current period in y_full
        """
        p = self.params
        equations = np.zeros(self.n_endo)
        
        # Current period values
        t_idx = period_t
        y_t = y_full[t_idx]
        y_tm1 = y_full[t_idx-1] if t_idx > 0 else np.zeros(self.n_endo)
        y_tm2 = y_full[t_idx-2] if t_idx > 1 else np.zeros(self.n_endo)
        y_tm3 = y_full[t_idx-3] if t_idx > 2 else np.zeros(self.n_endo)
        y_tp1 = y_full[t_idx+1] if t_idx < y_full.shape[0]-1 else np.zeros(self.n_endo)
        y_tp2 = y_full[t_idx+2] if t_idx < y_full.shape[0]-2 else np.zeros(self.n_endo)
        y_tp3 = y_full[t_idx+3] if t_idx < y_full.shape[0]-3 else np.zeros(self.n_endo)
        y_tp4 = y_full[t_idx+4] if t_idx < y_full.shape[0]-4 else np.zeros(self.n_endo)
        
        # Helper function to get variable by name
        def get_var(y_arr, name):
            return y_arr[self.endogenous_vars.index(name)]
        def get_exo(name):
            return x_t[self.exogenous_vars.index(name)]
        
        # Equation 0: IS Curve (hiato_prod)
        equations[0] = (get_var(y_t, 'hiato_prod') 
                       - p.beta_1 * get_var(y_tm1, 'hiato_prod')
                       + p.beta_2 * get_var(y_tm1, 'juros_real_gap') / 4
                       + p.beta_3 * get_exo('primario_governo')
                       - p.beta_4 * get_exo('hiato_mundial')
                       - get_var(y_t, 'shk_hiato_prod')
                       - get_exo('e_hiato_prod'))
        
        # Equation 1: Real interest rate gap (juros_real_gap)
        equations[1] = (get_var(y_t, 'juros_real_gap')
                       - get_var(y_t, 'exp_selic_focus_12m')
                       + get_var(y_t, 'exp_infl_focus_12m')
                       + get_var(y_t, 'juros_real_neutro'))
        
        # Equation 2: Expected Selic Focus 12m (exp_selic_focus_12m)
        equations[2] = (get_var(y_t, 'exp_selic_focus_12m')
                       - (0.5 * get_var(y_t, 'selic') 
                          + get_var(y_tp1, 'selic')
                          + get_var(y_tp2, 'selic')
                          + get_var(y_tp3, 'selic')
                          + 0.5 * get_var(y_tp4, 'selic')) / 4
                       - get_exo('e_selic_is'))
        
        # Equation 3: Expected inflation focus 12m (exp_infl_focus_12m)
        equations[3] = (get_var(y_t, 'exp_infl_focus_12m')
                       - p.phi_1 * get_var(y_tm1, 'exp_infl_focus_12m')
                       - p.phi_2 * get_var(y_t, 'exp_infl_consistente_modelo')
                       - p.phi_3 * get_var(y_tm1, 'infl_cheia_12m')
                       - (1 - p.phi_1 - p.phi_2 - p.phi_3) * get_var(y_t, 'meta_infl')
                       - get_exo('e_exp'))
        
        # Equation 4: Real neutral interest rate (juros_real_neutro)
        equations[4] = (get_var(y_t, 'juros_real_neutro')
                       - get_exo('juros_neutro_trend')
                       - get_var(y_t, 'juros_neutro_gap'))
        
        # Equation 5: Taylor Rule (selic)
        equations[5] = (get_var(y_t, 'selic')
                       - p.theta_1 * get_var(y_tm1, 'selic')
                       - p.theta_2 * get_var(y_tm2, 'selic')
                       - (1 - p.theta_1 - p.theta_2) * (
                           get_var(y_t, 'juros_neutro_taylor')
                           + get_var(y_t, 'meta_infl')
                           + p.theta_3 * (get_var(y_t, 'exp_infl_focus_12m') - get_var(y_t, 'meta_infl'))
                       )
                       - get_exo('e_taylor'))
        
        # Equation 6: Neutral interest rate gap (juros_neutro_gap)
        equations[6] = (get_var(y_t, 'juros_neutro_gap')
                       - get_var(y_tm1, 'juros_neutro_gap')
                       - get_exo('e_juros_neutro_gap'))
        
        # Equation 7: Shock to output gap (shk_hiato_prod)
        equations[7] = (get_var(y_t, 'shk_hiato_prod')
                       - p.beta_5 * get_var(y_tm1, 'shk_hiato_prod')
                       - get_exo('e_shk_hiato_prod'))
        
        # Equation 8: Full inflation quarterly (infl_cheia_tri)
        equations[8] = (get_var(y_t, 'infl_cheia_tri')
                       - (1 - p.w_adm) * get_var(y_t, 'infl_livres_tri')
                       - p.w_adm * get_var(y_t, 'infl_admin_tri'))
        
        # Equation 9: Free prices inflation quarterly (infl_livres_tri)
        equations[9] = (get_var(y_t, 'infl_livres_tri')
                       - p.alp_1_L * get_var(y_tm1, 'infl_livres_tri')
                       - p.alp_1_I * get_var(y_tm1, 'infl_cheia_12m') / 4
                       - (1 - p.alp_1_L - p.alp_1_I) * get_var(y_t, 'exp_infl_focus_12m') / 4
                       - p.alp_2 * get_var(y_t, 'desv_inf_importada')
                       - p.alp_3 * get_var(y_tm1, 'desv_cambio')
                       - p.alp_4 * get_var(y_t, 'hiato_prod')
                       - get_exo('e_phillips'))
        
        # Equation 10: Free prices inflation 12m (infl_livres_12m)
        equations[10] = (get_var(y_t, 'infl_livres_12m')
                        - get_var(y_t, 'infl_livres_tri')
                        - get_var(y_tm1, 'infl_livres_tri')
                        - get_var(y_tm2, 'infl_livres_tri')
                        - get_var(y_tm3, 'infl_livres_tri'))
        
        # Equation 11: Administered prices inflation 12m (infl_admin_12m)
        equations[11] = (get_var(y_t, 'infl_admin_12m')
                        - get_var(y_t, 'infl_admin_tri')
                        - get_var(y_tm1, 'infl_admin_tri')
                        - get_var(y_tm2, 'infl_admin_tri')
                        - get_var(y_tm3, 'infl_admin_tri'))
        
        # Equation 12: Full inflation 12m (infl_cheia_12m)
        equations[12] = (get_var(y_t, 'infl_cheia_12m')
                        - get_var(y_t, 'infl_cheia_tri')
                        - get_var(y_tm1, 'infl_cheia_tri')
                        - get_var(y_tm2, 'infl_cheia_tri')
                        - get_var(y_tm3, 'infl_cheia_tri'))
        
        # Equation 13: Deviation of imported inflation (desv_inf_importada)
        equations[13] = (get_var(y_t, 'desv_inf_importada')
                        - p.w_icbr_a * get_var(y_t, 'desv_icbr_a')
                        - p.w_icbr_m * get_var(y_t, 'desv_icbr_m')
                        - (1 - p.w_icbr_a - p.w_icbr_m) * get_var(y_t, 'desv_icbr_e'))
        
        # Equation 14: Exchange rate deviation (desv_cambio)
        equations[14] = (get_var(y_t, 'desv_cambio')
                        - get_var(y_t, 'var_cambio')
                        + get_var(y_t, 'var_ppc'))
        
        # Equations 15-17: ICBR deviations (desv_icbr_a, desv_icbr_m, desv_icbr_e)
        equations[15] = (get_var(y_t, 'desv_icbr_a') - get_var(y_t, 'icbr_a') + get_var(y_t, 'meta_infl'))
        equations[16] = (get_var(y_t, 'desv_icbr_m') - get_var(y_t, 'icbr_m') + get_var(y_t, 'meta_infl'))
        equations[17] = (get_var(y_t, 'desv_icbr_e') - get_var(y_t, 'icbr_e') + get_var(y_t, 'meta_infl'))
        
        # Equations 18-20: ICBR levels (icbr_a, icbr_m, icbr_e)
        equations[18] = (get_var(y_t, 'icbr_a') - get_var(y_t, 'var_cambio') - get_exo('e_icbr_a'))
        equations[19] = (get_var(y_t, 'icbr_m') - get_var(y_t, 'var_cambio') - get_exo('e_icbr_m'))
        equations[20] = (get_var(y_t, 'icbr_e') - get_var(y_t, 'var_cambio') - get_exo('e_icbr_e'))
        
        # Equation 21: Inflation target (meta_infl)
        equations[21] = (get_var(y_t, 'meta_infl') - p.meta_ss - get_exo('e_meta'))
        
        # Equation 22: UIP condition (var_cambio)
        equations[22] = (get_var(y_t, 'var_cambio')
                        - get_var(y_t, 'var_ppc')
                        + p.delta * (get_var(y_t, 'dif_juros') - get_var(y_tm1, 'dif_juros'))
                        - get_exo('e_uip'))
        
        # Equation 23: Taylor neutral rate (juros_neutro_taylor)
        equations[23] = (get_var(y_t, 'juros_neutro_taylor')
                        - get_exo('juros_neutro_trend')
                        - get_var(y_t, 'juros_neutro_taylor_gap'))
        
        # Equation 24: Taylor neutral rate gap (juros_neutro_taylor_gap)
        equations[24] = (get_var(y_t, 'juros_neutro_taylor_gap')
                        - get_var(y_tm1, 'juros_neutro_taylor_gap')
                        - get_exo('e_juros_neutro_taylor_gap'))
        
        # Equation 25: PPP variation (var_ppc)
        equations[25] = (get_var(y_t, 'var_ppc') 
                        - (get_var(y_t, 'meta_infl') - p.infl_ss_ext) / 4)
        
        # Equation 26: Interest rate differential (dif_juros)
        equations[26] = (get_var(y_t, 'dif_juros')
                        - (get_var(y_t, 'selic') - (get_exo('fed_funds') + get_exo('cds'))) / 4)
        
        # Equation 27: Model-consistent inflation expectations (exp_infl_consistente_modelo)
        equations[27] = (get_var(y_t, 'exp_infl_consistente_modelo')
                        - (0.5 * get_var(y_t, 'infl_cheia_12m')
                           + get_var(y_tp1, 'infl_cheia_12m')
                           + get_var(y_tp2, 'infl_cheia_12m')
                           + get_var(y_tp3, 'infl_cheia_12m')
                           + 0.5 * get_var(y_tp4, 'infl_cheia_12m')) / 4)
        
        # Equation 28: Brent oil price (brent)
        equations[28] = (get_var(y_t, 'brent')
                        - get_var(y_t, 'meta_infl')
                        - get_var(y_t, 'var_cambio')
                        - get_exo('e_brent'))
        
        # Equation 29: Administered prices inflation quarterly (infl_admin_tri)
        equations[29] = (get_var(y_t, 'infl_admin_tri')
                        - p.gama_1 * get_var(y_tm1, 'infl_cheia_12m') / 4
                        - p.gama_2 * get_var(y_t, 'brent')
                        - p.gama_3 * get_var(y_t, 'desv_cambio')
                        - p.gama_4 * get_var(y_tm1, 'desv_cambio'))
        
        # Equation 30: Accumulated exchange rate variation (var_cambio_acum)
        equations[30] = (get_var(y_t, 'var_cambio_acum')
                        - get_var(y_t, 'var_cambio')
                        - get_var(y_tm1, 'var_cambio')
                        - get_var(y_tm2, 'var_cambio')
                        - get_var(y_tm3, 'var_cambio'))
        
        return equations
    
    def solve_period(self, y_full, x_t, period_t):
        """Solve for one period using nonlinear solver"""
        def residual_func(y_t_guess):
            y_full[period_t] = y_t_guess
            return self.model_equations_full(y_full, x_t, period_t)
        
        # Initial guess (use previous period or zeros)
        if period_t > 0:
            initial_guess = y_full[period_t-1].copy()
        else:
            initial_guess = np.zeros(self.n_endo)
        
        # Solve
        try:
            solution = fsolve(residual_func, initial_guess, xtol=1e-8)
            return solution
        except:
            return initial_guess
    
    def simulate(self, periods: int = 2100, burnin: int = 100) -> pd.DataFrame:
        """Simulate the model with proper forward-looking expectations"""
        total_periods = periods + burnin
        buffer_periods = 8  # Extra periods for forward-looking terms
        extended_periods = total_periods + buffer_periods
        
        # Initialize arrays with buffer
        y_full = np.zeros((extended_periods, self.n_endo))
        x = np.zeros((extended_periods, self.n_exo))
        
        # Generate shocks
        np.random.seed(42)
        for i, var_name in enumerate(self.exogenous_vars):
            if var_name in self.shock_stds:
                x[:total_periods, i] = np.random.normal(0, self.shock_stds[var_name], total_periods)
        
        # Solve model period by period with iteration for forward-looking terms
        max_iterations = 5
        for iteration in range(max_iterations):
            y_old = y_full.copy()
            
            for t in range(total_periods):
                try:
                    y_full[t] = self.solve_period(y_full, x[t], t)
                except:
                    # If solver fails, use simple update
                    if t > 0:
                        y_full[t] = 0.9 * y_full[t-1]
            
            # Check convergence
            if np.max(np.abs(y_full - y_old)) < 1e-6:
                break
        
        # Remove burnin and buffer
        y_result = y_full[burnin:total_periods]
        
        # Create DataFrame
        df = pd.DataFrame(y_result, columns=self.endogenous_vars)
        df.index.name = 'Period'
        
        return df
    
    def compute_irfs(self, periods: int = 16, shock_size: float = 1.0) -> Dict[str, pd.DataFrame]:
        """Compute Impulse Response Functions"""
        irfs = {}
        buffer_periods = 8
        total_periods = periods + buffer_periods
        
        for shock_var in self.shock_stds.keys():
            print(f"Computing IRF for {shock_var}...")
            
            # Baseline (no shock)
            y_baseline = np.zeros((total_periods, self.n_endo))
            x_baseline = np.zeros((total_periods, self.n_exo))
            
            # Shock scenario
            y_shock = np.zeros((total_periods, self.n_endo))
            x_shock = np.zeros((total_periods, self.n_exo))
            shock_idx = self.exogenous_vars.index(shock_var)
            x_shock[0, shock_idx] = shock_size
            
            # Solve both scenarios
            for scenario_name, (y_sim, x_sim) in [("baseline", (y_baseline, x_baseline)), 
                                                  ("shock", (y_shock, x_shock))]:
                # Iterate for forward-looking expectations
                for iteration in range(3):
                    for t in range(periods):
                        try:
                            y_sim[t] = self.solve_period(y_sim, x_sim[t], t)
                        except:
                            if t > 0:
                                y_sim[t] = 0.95 * y_sim[t-1]
            
            # Compute IRF as difference
            irf_result = y_shock[:periods] - y_baseline[:periods]
            
            # Store IRF
            irf_df = pd.DataFrame(irf_result, columns=self.endogenous_vars)
            irf_df.index.name = 'Periods'
            irfs[shock_var] = irf_df
        
        return irfs
    
    def plot_irfs(self, irfs: Dict[str, pd.DataFrame], variables: List[str] = None):
        """Plot Impulse Response Functions"""
        if variables is None:
            variables = ['hiato_prod', 'infl_cheia_12m', 'selic', 'var_cambio']
        
        n_shocks = len(irfs)
        n_vars = len(variables)
        
        fig, axes = plt.subplots(n_vars, n_shocks, figsize=(4*n_shocks, 3*n_vars))
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        if n_shocks == 1:
            axes = axes.reshape(-1, 1)
        
        for j, (shock_name, irf_data) in enumerate(irfs.items()):
            for i, var in enumerate(variables):
                ax = axes[i, j]
                ax.plot(irf_data.index, irf_data[var], 'b-', linewidth=2)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax.set_title(f'{var} response to {shock_name}')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Periods')
                ax.set_ylabel(var)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    print("Initializing Bacen MSPP Model")
    model = BacenMSPPModel()
    
    print("Computing Impulse Response Functions...")
    irfs = model.compute_irfs(periods=16, shock_size=1.0)
    
    print("\nPlotting IRFs...")
    model.plot_irfs(irfs)
    
    # Show some IRF values
    print("\nSample IRF Values (e_uip shock):")
    if 'e_uip' in irfs:
        sample_vars = ['hiato_prod', 'infl_cheia_12m', 'selic', 'var_cambio']
        print(irfs['e_uip'][sample_vars].head(8))
    

    print("\nModel IRFs completed!")
