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

# Input Tanggal
start_date = st.sidebar.date_input("Mulai Tanggal", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("Sampai Tanggal", pd.to_datetime("today"))

# Tombol Eksekusi
if st.sidebar.button("🚀 Optimalkan Portofolio"):
    
    # --- 1. PROSES DATA ---
    with st.spinner('Sedang menarik data pasar...'):
        try:
            # Tarik data
            data = yf.download(tickers, start=start_date, end=end_date)['Close']
            
            # Hitung Log Returns (Harian)
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

    # [NEW] Function Hitung CVaR (Conditional Value at Risk)
    def calculate_cvar(returns, weights, alpha=0.05):
        # Hitung return harian portofolio: w1*r1 + w2*r2 ...
        portfolio_returns = (returns * weights).sum(axis=1)
        # Cari batas kerugian terburuk (misal 5% terburuk)
        cutoff = portfolio_returns.quantile(alpha)
        # Rata-rata dari kerugian di bawah batas itu
        cvar = portfolio_returns[portfolio_returns <= cutoff].mean()
        return cvar, portfolio_returns

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

    # Hitung CVaR
    cvar_val, portfolio_daily_ret = calculate_cvar(log_returns, optimal_weights)

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
        df_weights = df_weights[df_weights['Weight'] > 0.0001]
        
        fig_pie = px.pie(df_weights, values='Weight', names='Asset', 
                          title='Optimal Allocation', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.write("### Key Metrics")
        st.metric("Expected Annual Return", f"{opt_ret*100:.2f}%")
        st.metric("Annual Volatility (Risk)", f"{opt_std*100:.2f}%")
        st.metric("Sharpe Ratio", f"{opt_sharpe:.2f}")
        # [NEW] Tampilkan CVaR
        st.metric("CVaR (Worst 5% Loss)", f"{cvar_val*100:.2f}%", 
                  help="Conditional Value at Risk: Rata-rata kerugian saat pasar crash parah (Confidence Level 95%)")

    with col2:
        st.subheader("📈 Efficient Frontier")
        
        # Monte Carlo Simulation
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
        fig_ef.add_trace(go.Scatter(
            x=results[0,:], y=results[1,:], mode='markers',
            marker=dict(color=results[2,:], colorscale='Viridis', showscale=True, size=5, opacity=0.5),
            name='Random Portfolios'
        ))
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

    # --- 4. [NEW] BACKTESTING SECTION ---
    st.markdown("---")
    st.subheader("🔙 Historical Backtest (Simulation)")
    st.markdown("Bagaimana performa portofolio ini jika Anda membelinya sejak tanggal awal yang dipilih?")

    # Hitung Cumulative Return Portofolio
    # (1 + r_harian).cumprod() -> Pertumbuhan modal
    cum_return_portfolio = (1 + portfolio_daily_ret).cumprod()

    # Hitung Benchmark (Misal: 100% Bitcoin)
    if 'BTC-USD' in tickers:
        btc_ret = log_returns['BTC-USD']
        cum_return_btc = (1 + btc_ret).cumprod()
    else:
        # Kalau user gak pilih BTC, pake rata-rata pasar aja
        cum_return_btc = (1 + log_returns.mean(axis=1)).cumprod()

    # Gabung Dataframe buat Plotting
    df_backtest = pd.DataFrame({
        "Optimization Model": cum_return_portfolio,
        "Benchmark (BTC/Avg)": cum_return_btc
    })

    # Plot Line Chart
    fig_backtest = px.line(df_backtest, 
                           title='Growth of $1 Investment (Cumulative Return)',
                           labels={'value': 'Growth Multiplier', 'variable': 'Strategy'})
    st.plotly_chart(fig_backtest, use_container_width=True)

else:
    st.info("👈 Masukkan parameter di sidebar dan klik 'Optimalkan Portofolio' untuk memulai.")