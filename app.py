import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile, base64
import numpy as np

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

st.title("📊 Material Forecasting App For WCL")

# File upload
uploaded = st.file_uploader("Upload CSV", type=["csv"])

# Model selection
model_choice = st.selectbox(
    "Select Forecasting Model",
    ["Prophet", "ARIMA", "SARIMA", "ETS (Holt-Winters)"]
)

# Forecast horizon input
months_ahead = st.number_input("Forecast Horizon (months)", min_value=1, max_value=36, value=12)

if uploaded:
    # --- Read & Clean Data ---
    df = pd.read_csv(uploaded)
    df = df.dropna(axis=1, how="all")  # drop empty cols

    if "date" not in df.columns or "consumption" not in df.columns:
        st.error("❌ CSV must have 'date' and 'consumption' columns.")
    else:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.groupby("date")["consumption"].sum().reset_index()
        df = df.sort_values("date")

        st.write("### 📂 Cleaned Data", df.head())

        # --- Forecasting ---
        forecast_out = None
        total_forecast = 0
        fig = None

        # Prophet Model
        if model_choice == "Prophet":
            prophet_df = df.rename(columns={"date": "ds", "consumption": "y"})
            # Optional: add floor column if using logistic growth
            prophet_df['floor'] = 0

            # ✅ Correct Prophet initialization (no floor, no random_state)
            model = Prophet(growth='linear')
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=months_ahead, freq="M")
            forecast = model.predict(future)

            forecast_out = forecast[["ds", "yhat"]].tail(months_ahead).round(2)
            forecast_out = forecast_out.rename(columns={"ds": "date", "yhat": "forecast"})
            forecast_out["forecast"] = forecast_out["forecast"].clip(lower=0)
            total_forecast = forecast_out["forecast"].sum().round(2)

            fig, ax = plt.subplots(figsize=(10, 5))
            model.plot(forecast, ax=ax)

        # ARIMA Model
        elif model_choice == "ARIMA":
            model = ARIMA(df["consumption"], order=(1, 1, 1))
            fit = model.fit()
            forecast = fit.get_forecast(steps=months_ahead)
            forecast_out = pd.DataFrame({
                "date": pd.date_range(df["date"].max() + pd.offsets.MonthBegin(), periods=months_ahead, freq="M"),
                "forecast": forecast.predicted_mean.round(2)
            })
            forecast_out["forecast"] = forecast_out["forecast"].clip(lower=0)
            total_forecast = forecast_out["forecast"].sum().round(2)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["date"], df["consumption"], label="History")
            ax.plot(forecast_out["date"], forecast_out["forecast"], label="Forecast", color="red")
            ax.legend()

        # SARIMA Model
        elif model_choice == "SARIMA":
            model = SARIMAX(df["consumption"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            fit = model.fit(disp=False)
            forecast = fit.get_forecast(steps=months_ahead)
            forecast_out = pd.DataFrame({
                "date": pd.date_range(df["date"].max() + pd.offsets.MonthBegin(), periods=months_ahead, freq="M"),
                "forecast": forecast.predicted_mean.round(2)
            })
            forecast_out["forecast"] = forecast_out["forecast"].clip(lower=0)
            total_forecast = forecast_out["forecast"].sum().round(2)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["date"], df["consumption"], label="History")
            ax.plot(forecast_out["date"], forecast_out["forecast"], label="Forecast", color="red")
            ax.legend()

        # ETS Model
        elif model_choice == "ETS (Holt-Winters)":
            model = ExponentialSmoothing(df["consumption"], trend="add", seasonal="add", seasonal_periods=12)
            fit = model.fit()
            forecast_values = fit.forecast(months_ahead)
            forecast_values = forecast_values.clip(lower=0)
            forecast_out = pd.DataFrame({
                "date": pd.date_range(df["date"].max() + pd.offsets.MonthBegin(), periods=months_ahead, freq="M"),
                "forecast": forecast_values.round(2)
            })
            total_forecast = forecast_out["forecast"].sum().round(2)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["date"], df["consumption"], label="History")
            ax.plot(forecast_out["date"], forecast_out["forecast"], label="Forecast", color="red")
            ax.legend()

        # --- Show Forecast ---
        if forecast_out is not None:
            st.write("### 🔮 Forecast (Next Months)", forecast_out)
            st.pyplot(fig)

            # Executive Summary
            st.markdown("### 📑 Executive Summary")
            st.write(f"**Total forecasted demand for next {months_ahead} months = {total_forecast} units**")

            # --- PDF Export ---
            def create_pdf(dataframe, fig_path, total_forecast):
                styles = getSampleStyleSheet()
                pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                doc = SimpleDocTemplate(pdf_file.name, pagesize=A4)

                elements = []
                elements.append(Paragraph("📊 Forecast Report", styles["Title"]))
                elements.append(Spacer(1, 12))
                elements.append(Paragraph(
                    f"Executive Summary: Total forecasted demand for next {months_ahead} months = {total_forecast} units",
                    styles["Normal"]
                ))
                elements.append(Spacer(1, 12))

                # Forecast table
                table_data = [list(dataframe.columns)] + dataframe.values.tolist()
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]))
                elements.append(table)
                elements.append(Spacer(1, 12))

                # Add forecast plot
                elements.append(Paragraph("Forecast Plot:", styles["Heading2"]))
                img = Image(fig_path)
                img.drawHeight = 4*72  # 4 inches
                img.drawWidth = 6*72   # 6 inches
                elements.append(img)

                doc.build(elements)
                return pdf_file.name

            # Save matplotlib figure temporarily
            fig_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            fig.savefig(fig_path)

            if st.button("📥 Download Forecast PDF"):
                pdf_path = create_pdf(forecast_out, fig_path, total_forecast)
                with open(pdf_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="forecast_report.pdf">Download PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
footer = """
<style>
.footer {
    position: fixed;
    bottom: 0;
    width: 40%;
    text-align: center;
    color: white;
    font-size: 20px;
    padding: 10px 0;
}
</style>
<div class="footer">
    Designed & Built by Pratik Bihari
</div>
"""
st.markdown(footer, unsafe_allow_html=True)