# Crypto Portfolio Optimization 🚀📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Maintenance-Active-orange)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://crypto-portfolio-optimizer-mvqrj2y3de4xlcsxrpzswy.streamlit.app/)

## 📌 Gambaran Umum (Overview)
Repository ini berisi implementasi algoritma optimasi portofolio mata uang kripto (Cryptocurrency) menggunakan pendekatan **Modern Portfolio Theory (MPT)**.

Tujuan utama dari project ini adalah untuk mengkonstruksi portofolio aset kripto yang optimal dengan menyeimbangkan *risk* (risiko) dan *return* (imbal hasil), serta menghasilkan **Efficient Frontier**. Program ini membantu investor/trader menentukan alokasi bobot (weights) terbaik untuk setiap aset guna mencapai **Sharpe Ratio** maksimum.

## 🧮 Metodologi & Konsep Matematis
Project ini menerapkan konsep matematika terapan dan statistika, meliputi:
* **Logarithmic Returns:** Perhitungan return harian aset menggunakan log returns untuk normalitas data.
* **Covariance Matrix:** Mengukur korelasi antar aset kripto untuk diversifikasi risiko.
* **Sharpe Ratio:** Metrik utama untuk optimasi (Risk-adjusted return).
* **Efficient Frontier:** Visualisasi set portofolio optimal yang menawarkan return tertinggi untuk tingkat risiko tertentu.
* **Optimization Solver:** Menggunakan [Sebutkan metode, misal: SLSQP / Monte Carlo Simulation / Convex Optimization] untuk mencari bobot optimal.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Optimization:** SciPy (optimize), CVXPY (opsional)
* **Data Source:** [Sebutkan API, misal: yfinance / CCXT / CoinGecko API]
* **Visualization:** Matplotlib, Seaborn, Plotly

## ✨ Fitur Utama
1.  **Automated Data Fetching:** Mengambil data historis harga kripto (BTC, ETH, SOL, dll) secara otomatis.
2.  **Risk Analysis:** Menghitung volatilitas tahunan dan Expected Return.
3.  **Portfolio Allocation:** Output berupa persentase bobot ideal (misal: BTC 40%, ETH 30%, SOL 30%).
4.  **Visualisasi Data:** Plotting korelasi heatmap dan grafik Efficient Frontier.

## 🚀 Cara Menjalankan

### Prasyarat
Pastikan Python sudah terinstall. Install library yang dibutuhkan:

```bash
pip install pandas numpy matplotlib scipy yfinance
