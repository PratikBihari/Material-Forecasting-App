import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
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

# Page config
st.set_page_config(
    page_title="WCL Material Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1f4e79, #2e86ab);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}
.metric-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #2e86ab;
    margin: 0.5rem 0;
}
.stSelectbox > div > div {
    background-color: #000000;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 WCL Material Forecasting App</h1>
    <p>Advanced Time Series Forecasting for Material Consumption Planning</p>
    <p>Employing Cutting-Edge Machine Learning Tools </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File upload
    uploaded = st.file_uploader(
        "📁 Upload CSV File", 
        type=["csv"],
        help="Upload a CSV file with 'date' and 'consumption' columns"
    )
    
    st.divider()
    
    # Model selection
    model_choice = st.selectbox(
        "🤖 Select Forecasting Model",
        ["Prophet", "ARIMA", "SARIMA", "ETS (Holt-Winters)"],
        help="Choose the time series forecasting algorithm"
    )
    
    # Forecast horizon input
    months_ahead = st.number_input(
        "📅 Forecast Horizon (months)", 
        min_value=1, 
        max_value=36, 
        value=12,
        help="Number of months to forecast ahead"
    )
    
    # Model info
    model_info = {
        "Prophet": "Facebook's robust forecasting tool, handles seasonality well",
        "ARIMA": "Classic statistical model for stationary time series",
        "SARIMA": "ARIMA with seasonal components",
        "ETS (Holt-Winters)": "Exponential smoothing with trend and seasonality"
    }
    
    st.info(f"ℹ️ **{model_choice}**: {model_info[model_choice]}")

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

        # Data overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Records", len(df))
        with col2:
            st.metric("📅 Date Range", f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
        with col3:
            st.metric("📈 Avg Monthly Consumption", f"{df['consumption'].mean():.1f}")
        
        with st.expander("📂 View Data Sample"):
            st.dataframe(df.head(10), use_container_width=True)

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
            try:
                # Use auto ARIMA for better parameter selection
                from pmdarima import auto_arima
                auto_model = auto_arima(df["consumption"], seasonal=False, stepwise=True, suppress_warnings=True)
                model = ARIMA(df["consumption"], order=auto_model.order)
            except ImportError:
                # Fallback to manual ARIMA if pmdarima not available
                model = ARIMA(df["consumption"], order=(2, 1, 2))
            
            fit = model.fit()
            forecast = fit.get_forecast(steps=months_ahead)
            
            # Create proper date range starting from next month
            last_date = df["date"].max()
            start_date = last_date + pd.DateOffset(months=1)
            forecast_dates = pd.date_range(start=start_date, periods=months_ahead, freq='MS')
            
            forecast_out = pd.DataFrame({
                "date": forecast_dates,
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
            # Results section
            st.markdown("## 🔮 Forecast Results")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #333;">📊 Total Forecast</h3>
                    <h2 style="color: #2e86ab;">{:,.0f} units</h2>
                </div>
                """.format(total_forecast), unsafe_allow_html=True)
            
            with col2:
                avg_monthly = total_forecast / months_ahead
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #333;">📈 Avg Monthly</h3>
                    <h2 style="color: #2e86ab;">{:,.0f} units</h2>
                </div>
                """.format(avg_monthly), unsafe_allow_html=True)
            
            with col3:
                growth_rate = ((forecast_out['forecast'].mean() / df['consumption'].mean()) - 1) * 100
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #333;">📊 Growth Rate</h3>
                    <h2 style="color: {};">{}%</h2>
                </div>
                """.format("#28a745" if growth_rate >= 0 else "#dc3545", 
                          f"+{growth_rate:.1f}" if growth_rate >= 0 else f"{growth_rate:.1f}"), 
                unsafe_allow_html=True)
            
            # Chart and table in columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📈 Forecast Visualization")
                
                # Create interactive Plotly chart
                fig_plotly = go.Figure()
                
                # Add historical data
                fig_plotly.add_trace(go.Scatter(
                    x=df['date'], 
                    y=df['consumption'],
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Consumption:</b> %{y:,.0f} units<extra></extra>'
                ))
                
                # Add forecast data
                fig_plotly.add_trace(go.Scatter(
                    x=forecast_out['date'], 
                    y=forecast_out['forecast'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#ff7f0e', width=2),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> %{y:,.0f} units<extra></extra>'
                ))
                
                fig_plotly.update_layout(
                    title=f'{model_choice} Forecast - Next {months_ahead} Months',
                    xaxis_title='Date',
                    yaxis_title='Consumption',
                    hovermode='x unified',
                    showlegend=True,
                    height=400
                )
                
                st.plotly_chart(fig_plotly, use_container_width=True)
            
            with col2:
                st.subheader("📋 Forecast Table")
                forecast_display = forecast_out.copy()
                forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m')
                forecast_display['forecast'] = forecast_display['forecast'].apply(lambda x: f"{x:,.0f}")
                st.dataframe(forecast_display, use_container_width=True, hide_index=True)

            # --- PDF Export ---
            def create_pdf(dataframe, fig_path, total_forecast):
                from datetime import datetime
                from reportlab.lib.styles import ParagraphStyle
                from reportlab.lib.enums import TA_CENTER, TA_LEFT
                from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame
                from reportlab.lib.units import inch
                
                class NumberedCanvas:
                    def __init__(self, canvas, doc):
                        self.canvas = canvas
                        self.doc = doc
                    
                    def draw_page_number(self):
                        self.canvas.setFont("Helvetica", 9)
                        self.canvas.drawRightString(A4[0] - 0.75*inch, 0.75*inch, f"Page {self.canvas.getPageNumber()}")
                
                def add_page_number(canvas, doc):
                    numbered_canvas = NumberedCanvas(canvas, doc)
                    numbered_canvas.draw_page_number()
                
                styles = getSampleStyleSheet()
                pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                doc = BaseDocTemplate(pdf_file.name, pagesize=A4)
                
                frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
                template = PageTemplate(id='test', frames=frame, onPage=add_page_number)
                doc.addPageTemplates([template])

                # Custom styles
                title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=24, textColor=colors.darkblue, alignment=TA_CENTER, spaceAfter=20)
                subtitle_style = ParagraphStyle('CustomSubtitle', parent=styles['Heading2'], fontSize=16, textColor=colors.blue, alignment=TA_CENTER, spaceAfter=15)
                heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14, textColor=colors.darkgreen, spaceAfter=10)
                
                elements = []
                
                # Colorful Title Header
                elements.append(Paragraph("📊 MATERIAL FORECASTING REPORT", title_style))
                elements.append(Paragraph("Western Coalfields Limited", subtitle_style))
                elements.append(Paragraph("Material Consumption Analysis & Predictions", styles["Normal"]))
                elements.append(Spacer(1, 20))
                
                # Executive Summary (moved to top for impact)
                elements.append(Paragraph("📈 EXECUTIVE SUMMARY", heading_style))
                avg_monthly = total_forecast / months_ahead
                growth_rate = ((dataframe['forecast'].mean() / df['consumption'].mean()) - 1) * 100
                
                # Colorful summary boxes
                summary_data = [
                    ["Key Metrics", "Values"],
                    ["Total Forecasted Demand", f"{total_forecast:,.0f} units ({months_ahead} months)"],
                    ["Average Monthly Demand", f"{avg_monthly:,.0f} units"],
                    ["Growth Rate vs Historical", f"{growth_rate:+.1f}%"],
                    ["Peak Forecast Month", f"{dataframe.loc[dataframe['forecast'].idxmax(), 'date'].strftime('%Y-%m')} ({dataframe['forecast'].max():,.0f} units)"],
                    ["Minimum Forecast Month", f"{dataframe.loc[dataframe['forecast'].idxmin(), 'date'].strftime('%Y-%m')} ({dataframe['forecast'].min():,.0f} units)"]
                ]
                summary_table = Table(summary_data)
                summary_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.lightblue),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.lightblue, colors.lightyellow])
                ]))
                elements.append(summary_table)
                elements.append(Spacer(1, 15))
                
                # Forecast Visualization (moved up for visual impact)
                elements.append(Paragraph("📉 FORECAST VISUALIZATION", heading_style))
                img = Image(fig_path)
                img.drawHeight = 4*72
                img.drawWidth = 6*72
                elements.append(img)
                elements.append(Spacer(1, 15))
                
                # Monthly Forecast Details with colors
                elements.append(Paragraph("📅 MONTHLY FORECAST BREAKDOWN", heading_style))
                forecast_table_data = [["Month", "Forecasted Consumption", "Trend"]]
                for i, (_, row) in enumerate(dataframe.iterrows()):
                    if i == 0:
                        trend = "Baseline"
                    else:
                        prev_val = dataframe.iloc[i-1]['forecast']
                        curr_val = row['forecast']
                        trend = "↑ Increasing" if curr_val > prev_val else "↓ Decreasing" if curr_val < prev_val else "→ Stable"
                    forecast_table_data.append([row['date'].strftime('%Y-%m'), f"{row['forecast']:,.0f} units", trend])
                
                forecast_table = Table(forecast_table_data)
                forecast_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkgreen),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.lightgreen),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.lightgreen, colors.lightyellow])
                ]))
                elements.append(forecast_table)
                elements.append(Spacer(1, 15))
                
                # Statistical Comparison with colors
                elements.append(Paragraph("📊 STATISTICAL ANALYSIS", heading_style))
                # Safe variance calculation to avoid division by zero
                def safe_variance(forecast_val, historical_val):
                    if historical_val == 0:
                        return "N/A"
                    return f"{((forecast_val/historical_val-1)*100):+.1f}%"
                
                stats_data = [
                    ["Metric", "Historical Data", "Forecast Data", "Variance"],
                    ["Mean", f"{df['consumption'].mean():,.0f}", f"{dataframe['forecast'].mean():,.0f}", safe_variance(dataframe['forecast'].mean(), df['consumption'].mean())],
                    ["Median", f"{df['consumption'].median():,.0f}", f"{dataframe['forecast'].median():,.0f}", safe_variance(dataframe['forecast'].median(), df['consumption'].median())],
                    ["Maximum", f"{df['consumption'].max():,.0f}", f"{dataframe['forecast'].max():,.0f}", safe_variance(dataframe['forecast'].max(), df['consumption'].max())],
                    ["Minimum", f"{df['consumption'].min():,.0f}", f"{dataframe['forecast'].min():,.0f}", safe_variance(dataframe['forecast'].min(), df['consumption'].min())]
                ]
                stats_table = Table(stats_data)
                stats_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.purple),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.lavender),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(stats_table)
                elements.append(Spacer(1, 15))
                
                # Report Metadata with colors
                elements.append(Paragraph("📝 REPORT INFORMATION", heading_style))
                metadata_data = [
                    ["Parameter", "Details"],
                    ["Report Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    ["Forecasting Model", model_choice],
                    ["Forecast Horizon", f"{months_ahead} months"],
                    ["Historical Data Period", f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}"],
                    ["Total Data Points", f"{len(df)} records"]
                ]
                metadata_table = Table(metadata_data)
                metadata_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.orange),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.lightyellow),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(metadata_table)
                elements.append(Spacer(1, 15))
                
                # Model Information with colors
                model_descriptions = {
                    "Prophet": "Facebook's Prophet model handles seasonality and trends effectively, suitable for business forecasting.",
                    "ARIMA": "AutoRegressive Integrated Moving Average model, classic statistical approach for time series.",
                    "SARIMA": "Seasonal ARIMA model that accounts for seasonal patterns in the data.",
                    "ETS (Holt-Winters)": "Exponential smoothing model that captures trend and seasonal components."
                }
                
                elements.append(Paragraph("🤖 MODEL INFORMATION", heading_style))
                model_data = [
                    ["Model Details", "Description"],
                    ["Selected Model", model_choice],
                    ["Model Description", model_descriptions.get(model_choice, "Advanced forecasting model")],
                    ["Suitability", "Optimized for material consumption forecasting"]
                ]
                model_table = Table(model_data)
                model_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.teal),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.lightcyan),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(model_table)
                elements.append(Spacer(1, 20))
                
                # Colorful Footer
                footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=10, textColor=colors.darkblue, alignment=TA_CENTER)
                elements.append(Paragraph("🚀 Generated by WCL Material Forecasting System | Designed by Pratik Bihari 🚀", footer_style))

                doc.build(elements)
                return pdf_file.name

            # Save matplotlib figure temporarily
            fig_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            fig.savefig(fig_path)

            # Download section
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("📥 Download Forecast PDF", type="primary", use_container_width=True):
                    with st.spinner("Generating PDF report..."):
                        pdf_path = create_pdf(forecast_out, fig_path, total_forecast)
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="📄 Download Report",
                                data=f.read(),
                                file_name="forecast_report.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <p><strong>🚀 Designed & Built by Pratik Bihari</strong></p>
    <p>Material Forecasting Solution for Western Coalfields Limited</p>
</div>
""", unsafe_allow_html=True)