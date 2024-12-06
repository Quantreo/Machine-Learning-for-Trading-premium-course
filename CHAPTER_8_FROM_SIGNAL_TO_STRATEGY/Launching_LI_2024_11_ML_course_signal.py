from Strategies.LI_2024_11_ML_course_signal import *
from Quantreo.Backtest import *
from Quantreo.WalkForwardOptimization_bis import *

import warnings
warnings.filterwarnings("ignore")
from joblib import dump

# SAVE WEIGHTS
save = True
name = "LI_2024_11_MlStrategy"

# Import the data
df = pd.read_parquet("../Data_strategy/ML_Strategy_4H_EURUSD.parquet")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

training_data = pd.read_parquet("../Data_strategy/training_data_ML_course_strategy1.parquet").dropna().iloc[0:6000,:]
params_range = {
    "unique_path": [1]
}

params_fixed = {
    "tp": 0.005,
    "sl": -0.0025,
    "cost": 0.0002,
    "leverage": 5,
    "list_X": ['hurst', '0_to_20', '20_to_40', '60_to_80', '80_to_100', 'acceleration', 'spread', 'kama_diff',
               'kama_trend', 'autocorr_10', 'autocorr_50', 'ret_log_1', 'ret_log_5', 'ret_log_20', 'ret_log_50',
               'filling', 'amplitude', 'rolling_volatility_yang_zhang', 'linear_slope_6M','linear_slope_3M'],
    "train_mode": True,
    "training_data": training_data
}

# You can initialize the class into the variable RO, WFO or the name that you want (I put WFO for Walk forward Opti)
WFO = WalkForwardOptimization(df, MlCourseStrategy1, params_fixed, params_range,length_train_set=3_000, randomness=1.00)
WFO.run_optimization()

# Extract best parameters
params = WFO.best_params_smoothed[-1]
print("BEST PARAMETERS")
print(params)

model = params["model"]
sc = params["sc"]

# Save the weights
if save:
    dump(model, f"../models/saved/{name}_model.jolib")
    dump(sc, f"../models/saved/{name}_sc.jolib")

# Show the results
WFO.display()