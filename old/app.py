import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
import seaborn as sns
import os
st.set_page_config(layout="wide", page_title="Climate Anomaly Tracker")
st.title("Climate Seasonality and Anomaly Tracker")
st.markdown("This dashboard uses Fourier Transforms and Non-Linear Optimization to isolate weather anomalies in the Melbourne Minimum Temperatures dataset.")

@st.cache_data
def load_data():
    try:
        # Get the absolute path to the directory where app.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'melbourne_temp.csv')
        # Load the dataset, skipping any bad footer lines
        df = pd.read_csv(file_path, on_bad_lines='skip')
        
        # Rename columns to standardized names
        df.columns = ['Date', 'Temp']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Clean the 'Temp' column: remove non-numeric characters (like '?') and convert to float
        df['Temp'] = df['Temp'].astype(str).str.replace(r'[^0-9.]', '', regex=True)
        df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
        
        # Drop rows with NaN values
        df = df.dropna()
        df = df.set_index('Date')
        df = df.sort_index()
        
        # Ensure we have continuous days for FFT (fill missing dates if any exist)
        idx = pd.date_range(df.index.min(), df.index.max())
        df = df.reindex(idx)
        df['Temp'] = df['Temp'].interpolate(method='linear')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    st.sidebar.header("Dashboard Controls")
    
    # --- 1. Raw Data Visualization ---
    st.header("1. Raw Time-Series Data")
    fig_raw, ax_raw = plt.subplots(figsize=(12, 4))
    ax_raw.plot(df.index, df['Temp'], color='steelblue', linewidth=0.8)
    ax_raw.set_title("Daily Minimum Temperatures in Melbourne (1981-1990)")
    ax_raw.set_ylabel("Temperature (°C)")
    ax_raw.set_xlabel("Year")
    st.pyplot(fig_raw)
    
    # --- 2. Exploratory Data Analysis (EDA) ---
    st.header("2. Exploratory Data Analysis (EDA)")
    st.markdown("Comparing frequency distributions across different decades.")
    
    df['Year'] = df.index.year
    df['Decade'] = (df['Year'] // 10) * 10
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Kernel Density Estimate (KDE)")
        fig_kde, ax_kde = plt.subplots(figsize=(6, 4))
        sns.kdeplot(data=df, x='Temp', hue='Decade', fill=True, common_norm=False, palette='viridis', alpha=0.5, ax=ax_kde)
        ax_kde.set_title("Temperature Distribution by Decade")
        ax_kde.set_xlabel("Temperature (°C)")
        st.pyplot(fig_kde)
        
    with col2:
        st.subheader("Box Plot")
        fig_box, ax_box = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='Decade', y='Temp', palette='viridis', ax=ax_box)
        ax_box.set_title("Temperature Variance by Decade")
        ax_box.set_ylabel("Temperature (°C)")
        st.pyplot(fig_box)

    # --- 3. Fast Fourier Transform (FFT) ---
    st.header("3. Time-to-Frequency Domain (FFT)")
    
    # Get the temperature values as a NumPy array and mean-center them
    y = df['Temp'].values
    y_centered = y - np.mean(y)
    n = len(y)
    
    # Calculate FFT
    yf = fft(y_centered)
    # Calculate frequencies (cycles per day)
    xf = fftfreq(n, 1)
    
    # Get the positive frequencies and corresponding amplitudes
    pos_idx = np.where(xf > 0)
    freqs = xf[pos_idx]
    amplitudes = np.abs(yf[pos_idx])
    
    # Convert frequency from cycles/day to days/cycle (period)
    periods = 1 / freqs
    
    st.sidebar.subheader("FFT Filter")
    max_period_display = st.sidebar.slider("Max Period to Display (Days)", min_value=10, max_value=1000, value=500, step=10)
    
    fig_fft, ax_fft = plt.subplots(figsize=(12, 4))
    
    # Plot amplitude vs period
    valid_idx = periods <= max_period_display
    ax_fft.plot(periods[valid_idx], amplitudes[valid_idx], color='purple')
    ax_fft.set_title("Power Spectral Density (Period Domain)")
    ax_fft.set_xlabel("Period (Days)")
    ax_fft.set_ylabel("Amplitude")
    
    # Highlight the dominant frequency
    dominant_idx = np.argmax(amplitudes)
    dominant_period = periods[dominant_idx]
    ax_fft.axvline(x=dominant_period, color='red', linestyle='--', label=f'Dominant Period: {dominant_period:.2f} days')
    ax_fft.legend()
    st.pyplot(fig_fft)
    
    st.info(f"The FFT mathematically proves the strongest cyclical pattern repeats every **{dominant_period:.2f} days**, confirming the annual seasonal cycle.")

    # --- 4. Non-Linear Curve Optimization ---
    st.header("4. Levenberg-Marquardt Optimization")
    st.markdown("Fitting a mathematical sine wave model to the data using the dominant frequency extracted from the FFT.")
    
    # Define the sine wave function
    def sine_model(t, A, phase, C):
        # We fix omega using the dominant frequency found by FFT
        omega = 2 * np.pi / dominant_period 
        return A * np.sin(omega * t + phase) + C
    
    # Time array (0 to n-1)
    t = np.arange(n)
    
    st.sidebar.subheader("Optimization Guesses")
    guess_A = st.sidebar.number_input("Guess Amplitude (A)", value=5.0)
    guess_phase = st.sidebar.number_input("Guess Phase Shift", value=0.0)
    guess_C = st.sidebar.number_input("Guess Offset (C)", value=float(np.mean(y)))
    
    initial_guess = [guess_A, guess_phase, guess_C]
    
    try:
        # Run curve_fit
        popt, pcov = curve_fit(sine_model, t, y, p0=initial_guess)
        
        opt_A, opt_phase, opt_C = popt
        optimized_curve = sine_model(t, opt_A, opt_phase, opt_C)
        
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        col_opt1.metric("Optimized Amplitude (A)", f"{opt_A:.2f}")
        col_opt2.metric("Optimized Phase Shift", f"{opt_phase:.2f}")
        col_opt3.metric("Optimized Offset (C)", f"{opt_C:.2f}")
        
        # --- 5. Anomaly Isolation ---
        st.header("5. Anomaly Isolation")
        
        # Calculate residuals (raw data - optimized curve)
        residuals = y - optimized_curve
        std_dev = np.std(residuals)
        
        st.sidebar.subheader("Anomaly Threshold")
        z_threshold = st.sidebar.slider("Z-Score Threshold", min_value=1.5, max_value=4.0, value=2.5, step=0.1)
        
        # Flag anomalies where the residual > threshold * standard deviation
        upper_bound = optimized_curve + (z_threshold * std_dev)
        lower_bound = optimized_curve - (z_threshold * std_dev)
        
        anomalies_upper = y > upper_bound
        anomalies_lower = y < lower_bound
        
        fig_final, ax_final = plt.subplots(figsize=(14, 6))
        
        # Plot raw data (light grey)
        ax_final.plot(df.index, y, color='lightgrey', label='Raw Data', alpha=0.7)
        # Plot optimized curve
        ax_final.plot(df.index, optimized_curve, color='darkorange', linewidth=2, label='Optimized Baseline Model')
        # Plot bounds
        ax_final.plot(df.index, upper_bound, color='red', linestyle=':', alpha=0.5, label=f'+{z_threshold} Std Dev')
        ax_final.plot(df.index, lower_bound, color='blue', linestyle=':', alpha=0.5, label=f'-{z_threshold} Std Dev')
        
        # Scatter anomalies
        ax_final.scatter(df.index[anomalies_upper], y[anomalies_upper], color='red', s=20, label='Hot Anomalies', zorder=5)
        ax_final.scatter(df.index[anomalies_lower], y[anomalies_lower], color='blue', s=20, label='Cold Anomalies', zorder=5)
        
        ax_final.set_title(f"Melbourne Temperatures with Identified Anomalies (Z-Score > {z_threshold})")
        ax_final.set_ylabel("Temperature (°C)")
        ax_final.legend(loc='upper right')
        
        st.pyplot(fig_final)
        
        total_anomalies = np.sum(anomalies_upper) + np.sum(anomalies_lower)
        st.warning(f"**{total_anomalies}** weather anomalies detected mathematically based on current threshold.")

    except RuntimeError:
        st.error("Optimization failed to converge. Please adjust your initial guesses in the sidebar.")

