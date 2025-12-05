import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.graph_objects as go
import plotly.express as px

# --- 1. SETUP & TAMPILAN (UI) ---
st.set_page_config(page_title="Crypto Robo-Advisor", layout="wide")

# Custom CSS Modern
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }
    
    h1 { color: #00ADB5; font-weight: 700; }
    
    /* Card Metric Modern */
    div[data-testid="stMetric"] {
        background-color: rgba(34, 40, 49, 0.6);
        border: 1px solid rgba(0, 173, 181, 0.4);
        padding: 15px;
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }
    
    div[data-testid="stMetricValue"] { color: #00ADB5; }
    
    /* Button Styling */
    div.stButton > button {
        background-color: #00ADB5;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #00FFF5;
        color: #222831;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Crypto Robo-Advisor")
st.markdown("""
**Sistem cerdas manajemen portofolio Crypto.**
Pilih profil risiko Anda, dan biarkan algoritma matematika menentukan alokasi aset terbaik.
""")

# --- 2. SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Konfigurasi Portfolio")

risk_profile = st.sidebar.selectbox(
    "Pilih Profil Risiko Anda:",
    ("🐢 Konservatif (Cari Aman)", "⚖️ Moderat (Seimbang)", "🚀 Agresif (Cari Cuan Maksimal)")
)

st.sidebar.markdown("---")

default_tickers = "BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD"
tickers_input = st.sidebar.text_area("Daftar Aset (Ticker)", default_tickers)
tickers = [t.strip() for t in tickers_input.split(',')]

start_date = st.sidebar.date_input("Mulai Analisis", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("Sampai Tanggal", pd.to_datetime("today"))

# --- 3. LOGIC MATEMATIKA ---

def get_data(tickers, start, end):
    # Download data
    df = yf.download(tickers, start=start, end=end)
    
    # [FIX CRITICAL] Handle MultiIndex Columns (Penyebab utama data ketuker)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df['Close']
        except KeyError:
            df = df.xs('Close', level=0, axis=1)
            
    # Bersihkan Data Kosong
    df = df.dropna()
    
    # [WAJIB] Ambil urutan ticker yang ASLI dari Yahoo (Biasanya Abjad A-Z)
    actual_tickers = df.columns.tolist()
    
    return df, actual_tickers

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
    p_ret, p_var = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def minimize_volatility(weights, mean_returns, cov_matrix):
    p_ret, p_var = portfolio_performance(weights, mean_returns, cov_matrix)
    return p_var

def neg_return(weights, mean_returns, cov_matrix):
    p_ret, p_var = portfolio_performance(weights, mean_returns, cov_matrix)
    return -p_ret

def calculate_cvar(returns, weights, alpha=0.05):
    portfolio_returns = (returns * weights).sum(axis=1)
    cutoff = portfolio_returns.quantile(alpha)
    cvar = portfolio_returns[portfolio_returns <= cutoff].mean()
    return cvar, portfolio_returns

# --- 4. EKSEKUSI ---

if st.sidebar.button("💡 Generate Rekomendasi"):
    
    with st.spinner(f'Sedang meracik strategi {risk_profile}...'):
        try:
            # 1. Ambil Data (Outputnya sekarang DUA: Dataframe dan List Ticker yang Benar)
            df, tickers = get_data(tickers, start_date, end_date)
            
            # Hitung Returns
            log_returns = np.log(df / df.shift(1)).dropna()
            mean_returns = log_returns.mean() * 365
            cov_matrix = log_returns.cov() * 365
            
            num_assets = len(tickers)
            args = (mean_returns, cov_matrix)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for asset in range(num_assets))
            init_guess = num_assets * [1. / num_assets,]
            
            # 2. PILIH ALGORITMA
            if risk_profile == "🐢 Konservatif (Cari Aman)":
                result = sco.minimize(minimize_volatility, init_guess, args=args,
                                      method='SLSQP', bounds=bounds, constraints=constraints)
                st.info("ℹ️ Mode Konservatif: Algoritma fokus meminimalkan fluktuasi harga.")
                
            elif risk_profile == "🚀 Agresif (Cari Cuan Maksimal)":
                result = sco.minimize(neg_return, init_guess, args=args,
                                      method='SLSQP', bounds=bounds, constraints=constraints)
                st.warning("⚠️ Mode Agresif: Algoritma mengabaikan risiko demi return tertinggi.")
                
            else:
                result = sco.minimize(neg_sharpe_ratio, init_guess, args=args,
                                      method='SLSQP', bounds=bounds, constraints=constraints)
                st.success("✅ Mode Moderat: Mencari keseimbangan terbaik (Max Sharpe).")

            optimal_weights = result.x
            opt_ret, opt_std = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
            opt_sharpe = opt_ret / opt_std
            cvar_val, portfolio_daily_ret = calculate_cvar(log_returns, optimal_weights)

            # --- 5. VISUALISASI HASIL ---
            
            col1, col2 = st.columns([1.2, 2])
            
            with col1:
                st.subheader("🎯 Alokasi Aset")
                
                # PIE CHART
                df_weights = pd.DataFrame({'Asset': tickers, 'Weight': optimal_weights})
                df_weights = df_weights[df_weights['Weight'] > 0.0001]
                
                fig_pie = px.pie(df_weights, values='Weight', names='Asset', hole=0.5)
                fig_pie.update_layout(
                    legend=dict(orientation="h", y=-0.1),
                    margin=dict(t=0, b=0, l=0, r=0)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # [FITUR BARU] Tabel Ranking Return (Cek Fakta)
                st.write("### 📊 Ranking Kinerja Aset")
                st.caption("Aset dengan rata-rata return tertinggi (Data Historis):")
                ranking_df = pd.DataFrame({
                    'Aset': tickers,
                    'Annual Return': mean_returns.values * 100
                }).sort_values(by='Annual Return', ascending=False)
                st.dataframe(ranking_df.style.format({"Annual Return": "{:.2f}%"}), use_container_width=True)
                
                st.write("### Key Metrics")
                m1, m2 = st.columns(2)
                m1.metric("Return Tahunan", f"{opt_ret*100:.2f}%")
                m2.metric("Risiko (Volatilitas)", f"{opt_std*100:.2f}%")
                
                m3, m4 = st.columns(2)
                m3.metric("Sharpe Ratio", f"{opt_sharpe:.2f}")
                m4.metric("CVaR (Crash Risk)", f"{cvar_val*100:.2f}%", delta_color="inverse")

            with col2:
                # EFFICIENT FRONTIER
                st.subheader("📈 Efficient Frontier")
                
                num_portfolios = 2000 
                results = np.zeros((3, num_portfolios))
                for i in range(num_portfolios):
                    weights = np.random.random(num_assets)
                    weights /= np.sum(weights)
                    p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
                    results[0,i] = p_std
                    results[1,i] = p_ret
                    results[2,i] = p_ret / p_std

                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(
                    x=results[0,:], y=results[1,:], mode='markers',
                    marker=dict(color=results[2,:], colorscale='Viridis', size=4, opacity=0.5),
                    name='Random Portfolios'
                ))
                fig_ef.add_trace(go.Scatter(
                    x=[opt_std], y=[opt_ret], mode='markers',
                    marker=dict(color='red', size=20, symbol='star'),
                    name=f'Optimal ({risk_profile})'
                ))
                fig_ef.update_layout(xaxis_title='Risk (Volatility)', yaxis_title='Return', height=400)
                st.plotly_chart(fig_ef, use_container_width=True)
                
                # BACKTESTING
                st.markdown("---")
                st.subheader("🔙 Simulasi Backtest")
                
                cum_return_portfolio = (1 + portfolio_daily_ret).cumprod()
                
                # Logic Benchmark
                if 'BTC-USD' in tickers:
                    bench_data = log_returns['BTC-USD']
                    bench_label = "Benchmark (Bitcoin Only)"
                else:
                    bench_data = log_returns.mean(axis=1)
                    bench_label = "Market Average (Equal Weight)"
                
                cum_return_bench = (1 + bench_data).cumprod()
                
                df_backtest = pd.DataFrame({
                    "Model Robo-Advisor": cum_return_portfolio,
                    bench_label: cum_return_bench
                })
                
                fig_bt = px.line(df_backtest, title="Pertumbuhan Investasi $1")
                fig_bt.update_layout(height=400, xaxis_title="", yaxis_title="Multiplier")
                st.plotly_chart(fig_bt, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

else:
    st.info("👈 Silakan pilih Profil Risiko di sidebar dan klik tombol untuk memulai.")