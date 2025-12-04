import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.graph_objects as go
import plotly.express as px

# --- SETUP HALAMAN ---
st.set_page_config(page_title="Crypto Portfolio Optimizer", layout="wide")
st.title("💰 Crypto Portfolio Optimization (Markowitz Model)")
st.markdown("""
Aplikasi ini menggunakan **Modern Portfolio Theory (Mean-Variance Analysis)** untuk menentukan 
alokasi aset terbaik berdasarkan rasio *Risk vs Return* (Sharpe Ratio).
""")

# --- SIDEBAR: INPUT USER ---
st.sidebar.header("Konfigurasi Portofolio")

# Input Ticker
default_tickers = "BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD"
tickers_input = st.sidebar.text_area("Masukkan Ticker (pisahkan koma)", default_tickers)
tickers = [t.strip() for t in tickers_input.split(',')]

# Input Tanggal (Saran: Mulai 2021 biar dapet siklus Bull & Bear)
start_date = st.sidebar.date_input("Mulai Tanggal", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("Sampai Tanggal", pd.to_datetime("today"))

# Tombol Eksekusi
if st.sidebar.button("🚀 Optimalkan Portofolio"):
    
    # --- 1. PROSES DATA ---
    with st.spinner('Sedang menarik data pasar...'):
        try:
            # Tarik data
            data = yf.download(tickers, start=start_date, end=end_date)['Close']
            
            # Hitung Log Returns
            log_returns = np.log(data / data.shift(1)).dropna()
            
            # Annualize Mean & Covariance (365 hari crypto)
            mean_returns = log_returns.mean() * 365
            cov_matrix = log_returns.cov() * 365
            
            st.success("Data berhasil diproses!")
            
            # Tampilkan Raw Data (Expandable)
            with st.expander("Lihat Data Pergerakan Harga"):
                st.line_chart(data)

        except Exception as e:
            st.error(f"Gagal menarik data: {e}")
            st.stop()

    # --- 2. MODEL MATEMATIKA (The Math Engine) ---
    
    def portfolio_performance(weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return returns, std

    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
        p_ret, p_var = portfolio_performance(weights, mean_returns, cov_matrix)
        return -(p_ret - risk_free_rate) / p_var

    # Setup Optimizer
    num_assets = len(tickers)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    init_guess = num_assets * [1. / num_assets,]

    # Run Solver
    result = sco.minimize(neg_sharpe_ratio, init_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    opt_ret, opt_std = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    opt_sharpe = opt_ret / opt_std

    # --- 3. TAMPILKAN HASIL ---
    
    # Kolom Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🏆 Alokasi Terbaik")
        
        # Bikin Pie Chart pake Plotly
        df_weights = pd.DataFrame({
            'Asset': tickers,
            'Weight': optimal_weights
        })
        # Filter yang bobotnya kecil biar grafik bersih
        df_weights = df_weights[df_weights['Weight'] > 0.0001]
        
        fig_pie = px.pie(df_weights, values='Weight', names='Asset', 
                         title='Optimal Allocation', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.metric("Expected Annual Return", f"{opt_ret*100:.2f}%")
        st.metric("Annual Volatility (Risk)", f"{opt_std*100:.2f}%")
        st.metric("Sharpe Ratio", f"{opt_sharpe:.2f}")

    with col2:
        st.subheader("📈 Efficient Frontier")
        
        # Monte Carlo Simulation untuk Visualisasi
        num_portfolios = 3000
        results = np.zeros((3, num_portfolios))
        
        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
            results[0,i] = p_std
            results[1,i] = p_ret
            results[2,i] = p_ret / p_std # Sharpe

        # Plotly Scatter Plot
        fig_ef = go.Figure()
        
        # Titik-titik acak
        fig_ef.add_trace(go.Scatter(
            x=results[0,:], y=results[1,:], mode='markers',
            marker=dict(color=results[2,:], colorscale='Viridis', showscale=True, size=5, opacity=0.5),
            name='Random Portfolios'
        ))
        
        # Titik Optimal (Bintang)
        fig_ef.add_trace(go.Scatter(
            x=[opt_std], y=[opt_ret], mode='markers',
            marker=dict(color='red', size=20, symbol='star'),
            name='Optimal Portfolio'
        ))
        
        fig_ef.update_layout(
            title='Efficient Frontier (Risk vs Return)',
            xaxis_title='Risk (Volatility)',
            yaxis_title='Return',
            height=500
        )
        st.plotly_chart(fig_ef, use_container_width=True)

else:
    st.info("👈 Masukkan parameter di sidebar dan klik 'Optimalkan Portofolio' untuk memulai.")