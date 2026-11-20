import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv(r'D:\MSC_AI\Maths_AI\Climate_Tracker\melbourne_temp.csv', on_bad_lines='skip')
df.columns = ['Date', 'Temp']
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Temp'] = df['Temp'].astype(str).str.replace(r'[^0-9.]', '', regex=True)
df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
df = df.dropna()
df = df.set_index('Date')
df = df.sort_index()
idx = pd.date_range(df.index.min(), df.index.max())
df = df.reindex(idx)
df['Temp'] = df['Temp'].interpolate(method='linear')

print("=== DATASET STATS ===")
print(f"Records: {len(df)}")
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
print(f"Mean Temp: {df['Temp'].mean():.4f} C")
print(f"Std Dev:   {df['Temp'].std():.4f} C")
print(f"Min Temp:  {df['Temp'].min():.2f} C")
print(f"Max Temp:  {df['Temp'].max():.2f} C")

# FFT
y = df['Temp'].values
y_centered = y - np.mean(y)
n = len(y)
yf = fft(y_centered)
xf = fftfreq(n, 1)
pos_idx = np.where(xf > 0)
freqs = xf[pos_idx]
amplitudes = np.abs(yf[pos_idx])
periods = 1 / freqs
dominant_idx = np.argmax(amplitudes)
dominant_period = periods[dominant_idx]
print(f"\n=== FFT ===")
print(f"Dominant Period: {dominant_period:.4f} days")

# Curve fit
def sine_model(t, A, phase, C):
    omega = 2 * np.pi / dominant_period
    return A * np.sin(omega * t + phase) + C

t = np.arange(n)
popt, _ = curve_fit(sine_model, t, y, p0=[5.0, 0.0, float(np.mean(y))])
opt_A, opt_phase, opt_C = popt
optimized_curve = sine_model(t, opt_A, opt_phase, opt_C)
residuals_lm = y - optimized_curve
std_dev_lm = np.std(residuals_lm)
z_scores_lm = residuals_lm / std_dev_lm

print(f"\n=== LEVENBERG-MARQUARDT ===")
print(f"Amplitude A:  {opt_A:.4f}")
print(f"Phase:        {opt_phase:.4f}")
print(f"Offset C:     {opt_C:.4f}")
print(f"Residual Std: {std_dev_lm:.4f}")

# Z-Score anomalies
z_threshold = 2.5
anom_z_hot  = y > (optimized_curve + z_threshold * std_dev_lm)
anom_z_cold = y < (optimized_curve - z_threshold * std_dev_lm)
print(f"\n=== Z-SCORE (threshold={z_threshold}) ===")
print(f"Hot anomalies:  {np.sum(anom_z_hot)}")
print(f"Cold anomalies: {np.sum(anom_z_cold)}")
print(f"Total:          {np.sum(anom_z_hot)+np.sum(anom_z_cold)}")

# Seasonal Decomposition
decomp = seasonal_decompose(df['Temp'], model='additive', period=365, extrapolate_trend='freq')
df['Trend']    = decomp.trend
df['Seasonal'] = decomp.seasonal
df['Residual'] = decomp.resid

print(f"\n=== SEASONAL DECOMPOSITION ===")
print(f"Trend range:         {df['Trend'].max()-df['Trend'].min():.4f} C")
print(f"Seasonal amplitude:  {df['Seasonal'].max()-df['Seasonal'].min():.4f} C")
print(f"Residual std dev:    {df['Residual'].std():.4f} C")
print(f"Residual range:      {df['Residual'].min():.4f} to {df['Residual'].max():.4f} C")

# IQR on residuals
resid_valid = df['Residual'].dropna()
Q1  = resid_valid.quantile(0.25)
Q3  = resid_valid.quantile(0.75)
IQR = Q3 - Q1
iqr_upper = Q3 + 1.5 * IQR
iqr_lower = Q1 - 1.5 * IQR
anom_iqr_hot  = df['Residual'] > iqr_upper
anom_iqr_cold = df['Residual'] < iqr_lower
print(f"\n=== IQR METHOD ===")
print(f"Q1={Q1:.4f}, Q3={Q3:.4f}, IQR={IQR:.4f}")
print(f"Upper bound: {iqr_upper:.4f} C")
print(f"Lower bound: {iqr_lower:.4f} C")
print(f"Hot anomalies:  {anom_iqr_hot.sum()}")
print(f"Cold anomalies: {anom_iqr_cold.sum()}")
print(f"Total:          {anom_iqr_hot.sum()+anom_iqr_cold.sum()}")

# Isolation Forest (contamination=0.05)
df_if = df[['Temp','Residual']].dropna().copy()
X_if = df_if[['Temp','Residual']].values
iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
preds = iso.fit_predict(X_if)
df_if['IF_Anomaly'] = preds == -1
df_if['IF_Hot']  = df_if['IF_Anomaly'] & (df_if['Residual'] > 0)
df_if['IF_Cold'] = df_if['IF_Anomaly'] & (df_if['Residual'] < 0)
print(f"\n=== ISOLATION FOREST (contamination=0.05) ===")
print(f"Hot anomalies:  {df_if['IF_Hot'].sum()}")
print(f"Cold anomalies: {df_if['IF_Cold'].sum()}")
print(f"Total:          {df_if['IF_Anomaly'].sum()}")

# Consensus
z_hot   = pd.Series(anom_z_hot,  index=df.index)
z_cold  = pd.Series(anom_z_cold, index=df.index)
iq_hot  = anom_iqr_hot.reindex(df.index, fill_value=False)
iq_cold = anom_iqr_cold.reindex(df.index, fill_value=False)
if_hot  = df_if['IF_Hot'].reindex(df.index, fill_value=False)
if_cold = df_if['IF_Cold'].reindex(df.index, fill_value=False)
hot_count  = z_hot.astype(int)  + iq_hot.astype(int)  + if_hot.astype(int)
cold_count = z_cold.astype(int) + iq_cold.astype(int) + if_cold.astype(int)
consensus_hot  = hot_count  >= 2
consensus_cold = cold_count >= 2
print(f"\n=== CONSENSUS (>=2 methods) ===")
print(f"Consensus hot:   {consensus_hot.sum()}")
print(f"Consensus cold:  {consensus_cold.sum()}")
print(f"Any-method hot:  {(z_hot|iq_hot|if_hot).sum()}")
print(f"Any-method cold: {(z_cold|iq_cold|if_cold).sum()}")

# Monthly stats
df['Month_Name'] = df.index.strftime('%b')
monthly = df.groupby('Month_Name')['Temp'].agg(['mean','std','min','max'])
print(f"\n=== MONTHLY STATISTICS ===")
print(monthly.to_string())

# Seasonal stats
def get_season(m):
    if m in [12,1,2]: return 'Summer'
    elif m in [3,4,5]: return 'Autumn'
    elif m in [6,7,8]: return 'Winter'
    else: return 'Spring'
df['Season'] = df.index.month.map(get_season)
seasonal = df.groupby('Season')['Temp'].agg(['mean','std','min','max'])
print(f"\n=== SEASONAL STATISTICS ===")
print(seasonal.to_string())
print("\nDONE")
