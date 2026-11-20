import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
import io
import os

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Climate Anomaly Tracker")
st.title("🌡️ Climate Seasonality and Anomaly Tracker")
st.markdown(
    "This dashboard uses **Fourier Transforms** and **Non-Linear Optimization** to isolate "
    "weather anomalies in the Melbourne Minimum Temperatures dataset (1981–1990)."
)

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'melbourne_temp.csv')
        df = pd.read_csv(file_path, on_bad_lines='skip')
        df.columns = ['Date', 'Temp']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Temp'] = df['Temp'].astype(str).str.replace(r'[^0-9.]', '', regex=True)
        df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
        df = df.dropna()
        df = df.set_index('Date')
        df = df.sort_index()
        # Ensure continuous daily index for FFT
        idx = pd.date_range(df.index.min(), df.index.max())
        df = df.reindex(idx)
        df['Temp'] = df['Temp'].interpolate(method='linear')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


df = load_data()

if df is not None:
    # ─── Derived time columns ──────────────────────────────────────────────────
    df['Year']       = df.index.year
    df['Month']      = df.index.month
    df['Month_Name'] = df.index.strftime('%b')
    df['Decade']     = (df['Year'] // 10) * 10

    def get_season(month):
        # Southern-hemisphere seasons (Melbourne, Australia)
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:
            return 'Spring'

    df['Season'] = df['Month'].apply(get_season)

    # ─── Sidebar controls ─────────────────────────────────────────────────────
    st.sidebar.header("Dashboard Controls")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Raw Time-Series Data
    # ═══════════════════════════════════════════════════════════════════════════
    st.header("1. Raw Time-Series Data")
    fig_raw, ax_raw = plt.subplots(figsize=(12, 4))
    ax_raw.plot(df.index, df['Temp'], color='steelblue', linewidth=0.8)
    ax_raw.set_title("Daily Minimum Temperatures in Melbourne (1981–1990)")
    ax_raw.set_ylabel("Temperature (°C)")
    ax_raw.set_xlabel("Year")
    ax_raw.grid(alpha=0.3)
    st.pyplot(fig_raw)
    plt.close(fig_raw)

    # Quick summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{len(df):,}")
    m2.metric("Mean Temp (°C)", f"{df['Temp'].mean():.2f}")
    m3.metric("Min Temp (°C)", f"{df['Temp'].min():.1f}")
    m4.metric("Max Temp (°C)", f"{df['Temp'].max():.1f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Exploratory Data Analysis (ENHANCED)
    # ═══════════════════════════════════════════════════════════════════════════
    st.header("2. Exploratory Data Analysis (EDA)")
    st.markdown(
        "Exploring how temperature varies across **months**, **years**, and **seasons** "
        "within the 1981–1990 dataset."
    )

    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    season_order = ['Summer', 'Autumn', 'Winter', 'Spring']

    # --- Row 1: Monthly & Year-wise ---
    col_eda1, col_eda2 = st.columns(2)

    with col_eda1:
        st.subheader("Monthly Temperature Distribution")
        fig_month, ax_month = plt.subplots(figsize=(8, 4))
        sns.boxplot(
            data=df, x='Month_Name', y='Temp',
            order=month_order, palette='coolwarm', ax=ax_month
        )
        ax_month.set_title("Temperature Distribution by Month")
        ax_month.set_xlabel("Month")
        ax_month.set_ylabel("Temperature (°C)")
        ax_month.tick_params(axis='x', rotation=45)
        ax_month.grid(axis='y', alpha=0.3)
        st.pyplot(fig_month)
        st.caption(
            "Each box shows the median, interquartile range and outliers for that month. "
            "Winter months (Jun–Aug) show lower, tightly-clustered temperatures."
        )
        plt.close(fig_month)

    with col_eda2:
        st.subheader("Year-wise Temperature Comparison")
        fig_year, ax_year = plt.subplots(figsize=(8, 4))
        sns.boxplot(
            data=df, x='Year', y='Temp',
            palette='viridis', ax=ax_year
        )
        ax_year.set_title("Temperature Distribution by Year (1981–1990)")
        ax_year.set_xlabel("Year")
        ax_year.set_ylabel("Temperature (°C)")
        ax_year.tick_params(axis='x', rotation=45)
        ax_year.grid(axis='y', alpha=0.3)
        st.pyplot(fig_year)
        st.caption(
            "Year-wise comparison reveals inter-annual variability. "
            "Similar spread across years indicates a stable climate regime over this decade."
        )
        plt.close(fig_year)

    # --- Row 2: Seasonal KDE ---
    st.subheader("Seasonal Temperature Distribution")
    col_eda3, col_eda4 = st.columns(2)

    with col_eda3:
        fig_season_kde, ax_skde = plt.subplots(figsize=(8, 4))
        sns.kdeplot(
            data=df, x='Temp', hue='Season',
            hue_order=season_order,
            fill=True, common_norm=False,
            palette='Set2', alpha=0.5, ax=ax_skde
        )
        ax_skde.set_title("Temperature KDE by Season")
        ax_skde.set_xlabel("Temperature (°C)")
        ax_skde.set_ylabel("Density")
        ax_skde.grid(alpha=0.3)
        st.pyplot(fig_season_kde)
        st.caption(
            "The KDE curves show how the temperature probability density shifts dramatically "
            "between seasons. Summer and Winter distributions are well-separated."
        )
        plt.close(fig_season_kde)

    with col_eda4:
        fig_season_box, ax_sbox = plt.subplots(figsize=(8, 4))
        sns.boxplot(
            data=df, x='Season', y='Temp',
            order=season_order, palette='Set2', ax=ax_sbox
        )
        ax_sbox.set_title("Temperature Distribution by Season")
        ax_sbox.set_xlabel("Season")
        ax_sbox.set_ylabel("Temperature (°C)")
        ax_sbox.grid(axis='y', alpha=0.3)
        st.pyplot(fig_season_box)
        st.caption(
            "Winter (Jun–Aug) records the lowest median and narrowest spread. "
            "Summer (Dec–Feb) shows higher medians with more variability."
        )
        plt.close(fig_season_box)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — FFT  (PRESERVED AS-IS)
    # ═══════════════════════════════════════════════════════════════════════════
    st.header("3. Time-to-Frequency Domain (FFT)")

    y = df['Temp'].values
    y_centered = y - np.mean(y)
    n = len(y)

    yf = fft(y_centered)
    xf = fftfreq(n, 1)

    pos_idx = np.where(xf > 0)
    freqs = xf[pos_idx]
    amplitudes = np.abs(yf[pos_idx])
    periods = 1 / freqs

    st.sidebar.subheader("FFT Filter")
    max_period_display = st.sidebar.slider(
        "Max Period to Display (Days)", min_value=10, max_value=1000, value=500, step=10
    )

    fig_fft, ax_fft = plt.subplots(figsize=(12, 4))
    valid_idx = periods <= max_period_display
    ax_fft.plot(periods[valid_idx], amplitudes[valid_idx], color='purple')
    ax_fft.set_title("Power Spectral Density (Period Domain)")
    ax_fft.set_xlabel("Period (Days)")
    ax_fft.set_ylabel("Amplitude")
    ax_fft.grid(alpha=0.3)

    dominant_idx = np.argmax(amplitudes)
    dominant_period = periods[dominant_idx]
    ax_fft.axvline(
        x=dominant_period, color='red', linestyle='--',
        label=f'Dominant Period: {dominant_period:.2f} days'
    )
    ax_fft.legend()
    st.pyplot(fig_fft)
    plt.close(fig_fft)

    st.info(
        f"The FFT mathematically proves the strongest cyclical pattern repeats every "
        f"**{dominant_period:.2f} days**, confirming the annual seasonal cycle."
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Levenberg-Marquardt Optimization  (PRESERVED AS-IS)
    # ═══════════════════════════════════════════════════════════════════════════
    st.header("4. Levenberg-Marquardt Optimization")
    st.markdown(
        "Fitting a mathematical sine wave model to the data using the dominant frequency "
        "extracted from the FFT."
    )

    def sine_model(t, A, phase, C):
        omega = 2 * np.pi / dominant_period
        return A * np.sin(omega * t + phase) + C

    t = np.arange(n)

    st.sidebar.subheader("Optimization Guesses")
    guess_A     = st.sidebar.number_input("Guess Amplitude (A)", value=5.0)
    guess_phase = st.sidebar.number_input("Guess Phase Shift",   value=0.0)
    guess_C     = st.sidebar.number_input("Guess Offset (C)",    value=float(np.mean(y)))

    initial_guess = [guess_A, guess_phase, guess_C]

    try:
        popt, pcov = curve_fit(sine_model, t, y, p0=initial_guess)
        opt_A, opt_phase, opt_C = popt
        optimized_curve = sine_model(t, opt_A, opt_phase, opt_C)

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        col_opt1.metric("Optimized Amplitude (A)", f"{opt_A:.2f}")
        col_opt2.metric("Optimized Phase Shift",   f"{opt_phase:.2f}")
        col_opt3.metric("Optimized Offset (C)",    f"{opt_C:.2f}")

        # Residuals used by anomaly sections
        residuals_lm = y - optimized_curve          # Levenberg-Marquardt residuals
        std_dev_lm   = np.std(residuals_lm)
        z_scores_lm  = residuals_lm / std_dev_lm

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 5 — Seasonal Decomposition Analysis (NEW)
        # ═══════════════════════════════════════════════════════════════════════
        st.header("5. Seasonal Decomposition Analysis")
        st.markdown(
            r"""
Using the **additive decomposition model**:

$$Y(t) = T(t) + S(t) + R(t)$$

| Symbol | Meaning |
|--------|---------|
| $Y(t)$ | Original observed temperature |
| $T(t)$ | Trend component (long-term direction) |
| $S(t)$ | Seasonal component (repeating annual pattern) |
| $R(t)$ | Residual component (irregular / anomalous variation) |

Decomposition uses a period of **365 days** to capture the annual cycle.
"""
        )

        # Run decomposition — period=365 for annual seasonality
        decomp = seasonal_decompose(df['Temp'], model='additive', period=365, extrapolate_trend='period')

        df['Trend']    = decomp.trend
        df['Seasonal'] = decomp.seasonal
        df['Residual'] = decomp.resid

        fig_decomp, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig_decomp.suptitle("Seasonal Decomposition of Melbourne Minimum Temperatures", fontsize=14)

        axes[0].plot(df.index, df['Temp'],     color='steelblue',  linewidth=0.8)
        axes[0].set_ylabel("Temperature (°C)")
        axes[0].set_title("Original — Y(t): Raw daily minimum temperature")
        axes[0].grid(alpha=0.3)

        axes[1].plot(df.index, df['Trend'],    color='darkorange', linewidth=1.2)
        axes[1].set_ylabel("Temp (°C)")
        axes[1].set_title("Trend — T(t): Long-term warming or cooling direction")
        axes[1].grid(alpha=0.3)

        axes[2].plot(df.index, df['Seasonal'], color='green',      linewidth=0.8)
        axes[2].set_ylabel("Temp (°C)")
        axes[2].set_title("Seasonal — S(t): Annual repeating warm/cold cycle")
        axes[2].grid(alpha=0.3)

        axes[3].plot(df.index, df['Residual'], color='crimson',    linewidth=0.6, alpha=0.8)
        axes[3].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[3].set_ylabel("Residual (°C)")
        axes[3].set_title("Residual — R(t): Irregular deviation after removing trend & seasonality")
        axes[3].set_xlabel("Year")
        axes[3].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_decomp)
        plt.close(fig_decomp)

        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.metric("Trend Range (°C)", f"{df['Trend'].max() - df['Trend'].min():.2f}")
        col_d2.metric("Seasonal Amplitude (°C)", f"{df['Seasonal'].max() - df['Seasonal'].min():.2f}")
        col_d3.metric("Residual Std Dev (°C)", f"{df['Residual'].std():.2f}")

        st.caption(
            "**Trend**: Obtained by applying a moving average equal to the period (365 days). "
            "It captures gradual multi-year shifts in baseline temperature. "
            "**Seasonal**: The average pattern for each day-of-year after removing the trend. "
            "**Residual**: What remains after removing both trend and seasonality — "
            "large residuals indicate potential climate anomalies."
        )

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 6 — Anomaly Detection (ENHANCED — 3 methods)
        # ═══════════════════════════════════════════════════════════════════════
        st.header("6. Anomaly Detection")
        st.markdown(
            "Three complementary methods are used. The **residuals from seasonal decomposition** "
            "are used by the IQR and Isolation Forest methods so that normal seasonal variation "
            "is not misclassified as an anomaly."
        )

        # --- Sidebar controls for anomaly detection ---
        st.sidebar.subheader("Anomaly Thresholds")
        z_threshold = st.sidebar.slider(
            "Z-Score Threshold", min_value=1.5, max_value=4.0, value=2.5, step=0.1
        )
        contamination = st.sidebar.slider(
            "Isolation Forest Contamination", min_value=0.01, max_value=0.10, value=0.05, step=0.01
        )

        # ── Method A: Z-Score (PRESERVED from original) ──────────────────────
        st.subheader("Method A — Z-Score (Sine-Wave Residuals)")
        st.markdown(
            r"Flags observations where $|z| > \text{threshold}$, "
            r"where $z = \frac{r}{\sigma_r}$ and $r$ is the residual from the LM-optimized sine wave."
        )

        upper_bound_z = optimized_curve + (z_threshold * std_dev_lm)
        lower_bound_z = optimized_curve - (z_threshold * std_dev_lm)

        anom_z_hot  = y > upper_bound_z
        anom_z_cold = y < lower_bound_z

        fig_z, ax_z = plt.subplots(figsize=(14, 5))
        ax_z.plot(df.index, y,               color='lightgrey',  linewidth=0.8,  label='Raw Data',               alpha=0.7)
        ax_z.plot(df.index, optimized_curve, color='darkorange', linewidth=1.5,  label='Optimized Baseline')
        ax_z.plot(df.index, upper_bound_z,   color='red',        linestyle=':',  alpha=0.5, label=f'+{z_threshold}σ')
        ax_z.plot(df.index, lower_bound_z,   color='blue',       linestyle=':',  alpha=0.5, label=f'-{z_threshold}σ')
        ax_z.scatter(df.index[anom_z_hot],  y[anom_z_hot],  color='red',  s=18, label='Hot Anomalies',  zorder=5)
        ax_z.scatter(df.index[anom_z_cold], y[anom_z_cold], color='blue', s=18, label='Cold Anomalies', zorder=5)
        ax_z.set_title(f"Z-Score Anomaly Detection (Threshold = {z_threshold}σ)")
        ax_z.set_ylabel("Temperature (°C)")
        ax_z.set_xlabel("Year")
        ax_z.legend(loc='upper right')
        ax_z.grid(alpha=0.3)
        st.pyplot(fig_z)
        plt.close(fig_z)

        cz1, cz2, cz3 = st.columns(3)
        cz1.metric("🔴 Hot Anomalies (Z-Score)",  int(np.sum(anom_z_hot)))
        cz2.metric("🔵 Cold Anomalies (Z-Score)", int(np.sum(anom_z_cold)))
        cz3.metric("⚠️ Total (Z-Score)",           int(np.sum(anom_z_hot) + np.sum(anom_z_cold)))

        # ── Method B: IQR on decomposition residuals ─────────────────────────
        st.subheader("Method B — IQR (Seasonal Decomposition Residuals)")
        st.markdown(
            r"""
Applied to the **residual component** $R(t)$ from seasonal decomposition, so normal 
seasonal fluctuations are already removed.

$$\text{IQR} = Q_3 - Q_1$$
$$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR} \quad , \quad \text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$
"""
        )

        resid_valid = df['Residual'].dropna()
        Q1  = resid_valid.quantile(0.25)
        Q3  = resid_valid.quantile(0.75)
        IQR = Q3 - Q1
        iqr_upper = Q3 + 1.5 * IQR
        iqr_lower = Q1 - 1.5 * IQR

        # Anomaly mask aligned to full df index
        resid_series  = df['Residual']
        anom_iqr_hot  = resid_series > iqr_upper
        anom_iqr_cold = resid_series < iqr_lower

        fig_iqr, (ax_iqr1, ax_iqr2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        ax_iqr1.plot(df.index, df['Temp'],     color='steelblue', linewidth=0.8, label='Temperature')
        ax_iqr1.scatter(df.index[anom_iqr_hot],  df['Temp'][anom_iqr_hot],  color='red',  s=18, label='Hot Anomaly',  zorder=5)
        ax_iqr1.scatter(df.index[anom_iqr_cold], df['Temp'][anom_iqr_cold], color='blue', s=18, label='Cold Anomaly', zorder=5)
        ax_iqr1.set_ylabel("Temperature (°C)")
        ax_iqr1.set_title("IQR Anomaly Detection — Anomalies Flagged on Temperature Series")
        ax_iqr1.legend(); ax_iqr1.grid(alpha=0.3)

        ax_iqr2.plot(df.index, resid_series,   color='grey',   linewidth=0.6, alpha=0.8, label='Residual')
        ax_iqr2.axhline(iqr_upper, color='red',  linestyle='--', linewidth=1.2, label=f'Upper Bound ({iqr_upper:.2f}°C)')
        ax_iqr2.axhline(iqr_lower, color='blue', linestyle='--', linewidth=1.2, label=f'Lower Bound ({iqr_lower:.2f}°C)')
        ax_iqr2.axhline(0,         color='black',linestyle='-',  linewidth=0.8)
        ax_iqr2.scatter(df.index[anom_iqr_hot],  resid_series[anom_iqr_hot],  color='red',  s=18, zorder=5)
        ax_iqr2.scatter(df.index[anom_iqr_cold], resid_series[anom_iqr_cold], color='blue', s=18, zorder=5)
        ax_iqr2.set_ylabel("Residual (°C)")
        ax_iqr2.set_xlabel("Year")
        ax_iqr2.set_title("IQR Bounds Applied to Decomposition Residuals R(t)")
        ax_iqr2.legend(); ax_iqr2.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_iqr)
        plt.close(fig_iqr)

        ci1, ci2, ci3 = st.columns(3)
        ci1.metric("🔴 Hot Anomalies (IQR)",  int(np.sum(anom_iqr_hot)))
        ci2.metric("🔵 Cold Anomalies (IQR)", int(np.sum(anom_iqr_cold)))
        ci3.metric("⚠️ Total (IQR)",           int(np.sum(anom_iqr_hot) + np.sum(anom_iqr_cold)))

        st.caption(
            f"IQR = {IQR:.4f} °C  |  "
            f"Upper Bound = {iqr_upper:.4f} °C  |  "
            f"Lower Bound = {iqr_lower:.4f} °C"
        )

        # ── Method C: Isolation Forest ────────────────────────────────────────
        st.subheader("Method C — Isolation Forest")
        st.markdown(
            "Isolation Forest isolates anomalies by randomly partitioning the feature space. "
            "Points that require fewer splits to isolate are more likely to be anomalies. "
            "Features used: **Temperature** and **Decomposition Residual**."
        )

        # Build feature matrix — drop rows where Residual is NaN
        df_if = df[['Temp', 'Residual']].dropna().copy()
        X_if  = df_if[['Temp', 'Residual']].values

        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
        preds_if   = iso_forest.fit_predict(X_if)   # -1 = anomaly, 1 = normal

        df_if['IF_Anomaly'] = preds_if == -1
        # Hot/cold: determined by the sign of the residual
        df_if['IF_Hot']  = df_if['IF_Anomaly'] & (df_if['Residual'] > 0)
        df_if['IF_Cold'] = df_if['IF_Anomaly'] & (df_if['Residual'] < 0)

        fig_if, ax_if = plt.subplots(figsize=(14, 5))
        ax_if.plot(df_if.index, df_if['Temp'], color='lightgrey', linewidth=0.8, label='Temperature', alpha=0.7)
        ax_if.scatter(
            df_if.index[df_if['IF_Hot']],  df_if['Temp'][df_if['IF_Hot']],
            color='red',  s=18, label='Hot Anomaly',  zorder=5
        )
        ax_if.scatter(
            df_if.index[df_if['IF_Cold']], df_if['Temp'][df_if['IF_Cold']],
            color='blue', s=18, label='Cold Anomaly', zorder=5
        )
        ax_if.set_title(f"Isolation Forest Anomaly Detection (Contamination = {contamination:.0%})")
        ax_if.set_ylabel("Temperature (°C)")
        ax_if.set_xlabel("Year")
        ax_if.legend(loc='upper right')
        ax_if.grid(alpha=0.3)
        st.pyplot(fig_if)
        plt.close(fig_if)

        cif1, cif2, cif3 = st.columns(3)
        cif1.metric("🔴 Hot Anomalies (IF)",  int(df_if['IF_Hot'].sum()))
        cif2.metric("🔵 Cold Anomalies (IF)", int(df_if['IF_Cold'].sum()))
        cif3.metric("⚠️ Total (IF)",           int(df_if['IF_Anomaly'].sum()))

        # ── Multi-method comparison visualization ─────────────────────────────
        st.subheader("Multi-Method Comparison")
        st.markdown("Dates flagged by **2 or more** methods are shown as consensus anomalies.")

        # Build boolean series for each method (aligned to full df index)
        z_hot_series    = pd.Series(anom_z_hot,  index=df.index)
        z_cold_series   = pd.Series(anom_z_cold, index=df.index)
        iqr_hot_series  = anom_iqr_hot.reindex(df.index, fill_value=False)
        iqr_cold_series = anom_iqr_cold.reindex(df.index, fill_value=False)
        if_hot_series   = df_if['IF_Hot'].reindex(df.index,  fill_value=False)
        if_cold_series  = df_if['IF_Cold'].reindex(df.index, fill_value=False)

        any_hot  = z_hot_series  | iqr_hot_series  | if_hot_series
        any_cold = z_cold_series | iqr_cold_series | if_cold_series

        # Count of methods agreeing on hot/cold per day
        hot_count  = z_hot_series.astype(int)  + iqr_hot_series.astype(int)  + if_hot_series.astype(int)
        cold_count = z_cold_series.astype(int) + iqr_cold_series.astype(int) + if_cold_series.astype(int)
        consensus_hot  = hot_count  >= 2
        consensus_cold = cold_count >= 2

        fig_comp, ax_comp = plt.subplots(figsize=(14, 5))
        ax_comp.plot(df.index, df['Temp'], color='lightgrey', linewidth=0.8, label='Temperature', alpha=0.6)
        ax_comp.scatter(df.index[any_hot],       df['Temp'][any_hot],       color='salmon',    s=12, alpha=0.5, label='Hot — any method')
        ax_comp.scatter(df.index[any_cold],      df['Temp'][any_cold],      color='lightblue', s=12, alpha=0.5, label='Cold — any method')
        ax_comp.scatter(df.index[consensus_hot],  df['Temp'][consensus_hot],  color='darkred',   s=35, zorder=5,  label='Hot — consensus (≥2)')
        ax_comp.scatter(df.index[consensus_cold], df['Temp'][consensus_cold], color='darkblue',  s=35, zorder=5,  label='Cold — consensus (≥2)')
        ax_comp.set_title("Multi-Method Anomaly Comparison")
        ax_comp.set_ylabel("Temperature (°C)")
        ax_comp.set_xlabel("Year")
        ax_comp.legend(loc='upper right', fontsize=8)
        ax_comp.grid(alpha=0.3)
        st.pyplot(fig_comp)
        plt.close(fig_comp)

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Consensus Hot",   int(consensus_hot.sum()))
        cc2.metric("Consensus Cold",  int(consensus_cold.sum()))
        cc3.metric("Any-method Hot",  int(any_hot.sum()))
        cc4.metric("Any-method Cold", int(any_cold.sum()))

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 7 — Anomaly Investigation (NEW)
        # ═══════════════════════════════════════════════════════════════════════
        st.header("7. Anomaly Investigation")
        st.markdown(
            "Filter, explore, and export all detected anomalies from the three detection methods."
        )

        # ── Build unified anomaly records table ───────────────────────────────
        records = []

        def build_records(mask_hot, mask_cold, method_label):
            """Create a list of dicts for hot and cold anomalies from a boolean mask."""
            for idx_val in df.index[mask_hot]:
                temp_val   = df.loc[idx_val, 'Temp']
                resid_val  = df.loc[idx_val, 'Residual'] if pd.notna(df.loc[idx_val, 'Residual']) else np.nan
                z_val      = z_scores_lm[df.index.get_loc(idx_val)]
                exp_temp   = temp_val - resid_val if pd.notna(resid_val) else np.nan
                records.append({
                    'Date':              idx_val.date(),
                    'Actual_Temp':       round(temp_val, 2),
                    'Expected_Temp':     round(exp_temp, 2) if pd.notna(exp_temp) else np.nan,
                    'Residual':          round(resid_val, 4) if pd.notna(resid_val) else np.nan,
                    'Z_Score':           round(float(z_val), 4),
                    'Classification':    'Hot',
                    'Detection_Method':  method_label,
                })
            for idx_val in df.index[mask_cold]:
                temp_val   = df.loc[idx_val, 'Temp']
                resid_val  = df.loc[idx_val, 'Residual'] if pd.notna(df.loc[idx_val, 'Residual']) else np.nan
                z_val      = z_scores_lm[df.index.get_loc(idx_val)]
                exp_temp   = temp_val - resid_val if pd.notna(resid_val) else np.nan
                records.append({
                    'Date':              idx_val.date(),
                    'Actual_Temp':       round(temp_val, 2),
                    'Expected_Temp':     round(exp_temp, 2) if pd.notna(exp_temp) else np.nan,
                    'Residual':          round(resid_val, 4) if pd.notna(resid_val) else np.nan,
                    'Z_Score':           round(float(z_val), 4),
                    'Classification':    'Cold',
                    'Detection_Method':  method_label,
                })

        build_records(anom_z_hot,  anom_z_cold,  'Z-Score')
        build_records(anom_iqr_hot.reindex(df.index, fill_value=False),
                      anom_iqr_cold.reindex(df.index, fill_value=False), 'IQR')
        build_records(if_hot_series.reindex(df.index, fill_value=False),
                      if_cold_series.reindex(df.index, fill_value=False), 'Isolation Forest')

        anom_df = pd.DataFrame(records)
        anom_df['Date'] = pd.to_datetime(anom_df['Date'])
        anom_df = anom_df.sort_values('Date').reset_index(drop=True)

        # ── Sidebar filters ────────────────────────────────────────────────────
        st.sidebar.subheader("Anomaly Investigation Filters")

        date_min = anom_df['Date'].min().date()
        date_max = anom_df['Date'].max().date()

        filter_start = st.sidebar.date_input("Start Date", value=date_min,
                                             min_value=date_min, max_value=date_max)
        filter_end   = st.sidebar.date_input("End Date",   value=date_max,
                                             min_value=date_min, max_value=date_max)

        all_years    = sorted(anom_df['Date'].dt.year.unique())
        filter_years = st.sidebar.multiselect("Filter by Year", options=all_years, default=all_years)

        filter_hot  = st.sidebar.checkbox("Show Hot Anomalies",  value=True)
        filter_cold = st.sidebar.checkbox("Show Cold Anomalies", value=True)

        all_methods   = anom_df['Detection_Method'].unique().tolist()
        filter_method = st.sidebar.multiselect(
            "Detection Method", options=all_methods, default=all_methods
        )

        # ── Apply filters ──────────────────────────────────────────────────────
        mask_date  = (anom_df['Date'].dt.date >= filter_start) & (anom_df['Date'].dt.date <= filter_end)
        mask_year  = anom_df['Date'].dt.year.isin(filter_years)
        mask_class = (
            (anom_df['Classification'] == 'Hot')  if (filter_hot  and not filter_cold) else
            (anom_df['Classification'] == 'Cold') if (filter_cold and not filter_hot)  else
            pd.Series(True, index=anom_df.index)  if (filter_hot  and filter_cold)      else
            pd.Series(False, index=anom_df.index)
        )
        mask_method = anom_df['Detection_Method'].isin(filter_method)

        filtered_df = anom_df[mask_date & mask_year & mask_class & mask_method].copy()
        filtered_df['Date'] = filtered_df['Date'].dt.strftime('%Y-%m-%d')

        # ── Display table ──────────────────────────────────────────────────────
        st.markdown(f"**{len(filtered_df)} anomaly records** match the current filters.")

        def highlight_class(row):
            color = '#ffcccc' if row['Classification'] == 'Hot' else '#cce0ff'
            return [f'background-color: {color}'] * len(row)

        st.dataframe(
            filtered_df.style.apply(highlight_class, axis=1),
            use_container_width=True,
            height=400
        )

        # ── CSV Download ───────────────────────────────────────────────────────
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Filtered Anomalies as CSV",
            data=csv_buffer.getvalue(),
            file_name="anomaly_records.csv",
            mime="text/csv"
        )

    except RuntimeError as e:
        st.error(f"Optimization failed to converge. Please adjust initial guesses in the sidebar. ({e})")
