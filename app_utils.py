# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Page config
st.set_page_config(
    page_title="Pipeline Pilferage Detection",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛢️ Pipeline Pilferage Detection Dashboard")
st.markdown("---")

# File upload
col1, col2 = st.columns(2)
with col1:
    pidws_file = st.file_uploader("Upload PIDWS data (df_pidws_III.xlsx)", type=['xlsx'])
with col2:
    lds_file = st.file_uploader("Upload LDS data (df_lds_III.xlsx)", type=['xlsx'])

if pidws_file is not None and lds_file is not None:
    # Load data
    @st.cache_data
    def load_data(pidws_buffer, lds_buffer):
        df_pidws = pd.read_excel(pidws_buffer)
        df_lds = pd.read_excel(lds_buffer)
        return df_pidws, df_lds
    
    df_pidws, df_lds = load_data(pidws_file, lds_file)
    
    st.success(f"✅ Loaded {len(df_pidws)} PIDWS records and {len(df_lds)} LDS records")
    
    # Sidebar controls
    st.sidebar.header("📊 Analysis Parameters")
    chainage_tol = st.sidebar.slider("Chainage Tolerance (km)", 0.1, 2.0, 0.5, 0.1)
    time_window_hours = st.sidebar.slider("Time Window (hours)", 12, 72, 48, 6)
    
    # Parse functions (same as original)
    @st.cache_data
    def parse_pidws(df):
        df = df.copy()
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d-%m-%Y %H:%M:%S')
        
        def parse_duration(dur_str):
            if pd.isna(dur_str):
                return pd.Timedelta(0)
            dur_str = str(dur_str).strip().lower().replace(' ', '')
            mins, secs = 0, 0
            if 'm' in dur_str:
                m_part = dur_str.split('m')[0]
                if m_part.isdigit():
                    mins = int(m_part)
                dur_str = dur_str.split('m')[1]
            if 's' in dur_str:
                s_part = dur_str.replace('s', '')
                if s_part.isdigit():
                    secs = int(s_part)
            return pd.Timedelta(minutes=mins, seconds=secs)
        
        df['duration_td'] = df['Event Duration'].apply(parse_duration)
        df['end_time'] = df['DateTime'] + df['duration_td']
        return df
    
    @st.cache_data
    def parse_lds(df):
        df = df.copy()
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])
        return df
    
    df_pidws = parse_pidws(df_pidws)
    df_lds = parse_lds(df_lds)
    
    # Classification
    @st.cache_data
    def classify_pilferage(pidws_df, lds_df, chainage_tol, time_window_hours):
        classified = []
        for _, event in pidws_df.iterrows():
            window_end = event['end_time'] + pd.Timedelta(hours=time_window_hours)
            mask = (lds_df['DateTime'] > window_end) & \
                   (np.abs(lds_df['chainage'] - event['chainage']) <= chainage_tol)
            matches = lds_df[mask].copy()
            if not matches.empty:
                matches['linked_event_time'] = event['DateTime']
                matches['linked_chainage'] = event['chainage']
                matches['pilferage_score'] = 1 / (1 + (matches['DateTime'] - window_end).dt.total_seconds() / 3600)
                classified.append(matches)
        
        if classified:
            return pd.concat(classified, ignore_index=True)
        return pd.DataFrame()
    
    with st.spinner("🔍 Classifying pilferage events..."):
        pilferage_leaks = classify_pilferage(df_pidws, df_lds, chainage_tol, time_window_hours)
    
    # Classification summary
    if not pilferage_leaks.empty:
        df_lds_classified = df_lds.copy()
        df_lds_classified['is_pilferage'] = False
        
        pilferage_ids = pilferage_leaks[['DateTime', 'chainage']].drop_duplicates()
        mask_pilferage = df_lds_classified.set_index(['DateTime', 'chainage']).index.isin(
            pilferage_ids.set_index(['DateTime', 'chainage']).index
        )
        df_lds_classified.loc[mask_pilferage, 'is_pilferage'] = True
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total LDS Records", len(df_lds))
        col2.metric("Pilferage Events", len(pilferage_leaks), delta=f"{len(pilferage_leaks)/len(df_lds)*100:.1f}%")
        col3.metric("Avg Pilferage Score", f"{pilferage_leaks['pilferage_score'].mean():.3f}")
        col4.metric("Top Chainage", f"{pilferage_leaks['linked_chainage'].mean():.1f} km")
    else:
        st.warning("⚠️ No pilferage events detected with current parameters")
    
    st.markdown("---")
    
    # Visualizations (2x2 grid matching original)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Chainage Distribution', 'Temporal Pattern', 'Leak Size: Pilferage vs Others', 'Leaks Timeline'),
        specs=[[{"type": "histogram"}, {"type": "xy"}],
               [{"type": "box"}, {"type": "scatter"}]]
    )
    
    # 1. Chainage distribution
    fig.add_trace(
        go.Histogram(x=df_pidws['chainage'], name='PIDWS Digging', opacity=0.7, marker_color='orange'),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=df_lds['chainage'], name='LDS Leaks', opacity=0.7, marker_color='blue'),
        row=1, col=1
    )
    if not pilferage_leaks.empty:
        fig.add_vline(
            x=pilferage_leaks['linked_chainage'].mean(),
            line_dash="dash", line_color="red", annotation_text="Pilferage Mean",
            row=1, col=1
        )
    
    # 2. Time series
    all_events = pd.concat([
        df_pidws[['DateTime', 'chainage']].assign(type='Digging'),
        df_lds[['DateTime', 'chainage']].assign(type='Leak'),
    ], ignore_index=True)
    if not pilferage_leaks.empty:
        all_events = pd.concat([
            all_events,
            pilferage_leaks[['DateTime', 'linked_chainage']].rename(columns={'linked_chainage':'chainage'}).assign(type='Pilferage')
        ], ignore_index=True)
    
    time_counts = all_events.groupby([all_events['DateTime'].dt.floor('H'), 'type']).size().unstack(fill_value=0)
    for col in time_counts.columns:
        fig.add_trace(
            go.Scatter(x=time_counts.index, y=time_counts[col], mode='lines', name=col),
            row=1, col=2
        )
    
    # 3. Leak size boxplot
    if not pilferage_leaks.empty:
        box_data = df_lds_classified[['leak size', 'is_pilferage']].copy()
        box_data['classification'] = box_data['is_pilferage'].map({True: 'Pilferage', False: 'Other'})
        fig.add_trace(
            go.Box(x=box_data['classification'], y=box_data['leak size'], name='Leak Size'),
            row=2, col=1
        )
    
    # 4. Timeline scatter
    if not pilferage_leaks.empty:
        colors = ['red' if x else 'blue' for x in df_lds_classified['is_pilferage']]
        fig.add_trace(
            go.Scatter(
                x=df_lds_classified['DateTime'], y=df_lds_classified['chainage'],
                mode='markers', marker=dict(color=colors, size=6, opacity=0.6),
                name='Leaks', showlegend=False
            ),
            row=2, col=2
        )
        fig.add_annotation(
            text="🔴 Pilferage, 🔵 Other", xref="paper", yref="paper",
            x=0.05, y=0.95, showarrow=False, row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="Pilferage Detection Analysis", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Chainage clusters table
    st.markdown("### 📍 Top Chainage Clusters")
    if not pilferage_leaks.empty:
        clusters = pilferage_leaks.groupby('linked_chainage')['leak size'].agg(['count', 'mean', 'max']).round(1)
        st.dataframe(clusters, use_container_width=True)
    
    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        csv_buffer = io.StringIO()
        if not pilferage_leaks.empty:
            df_lds_classified.to_csv(csv_buffer, index=False)
            st.download_button(
                "💾 Download Classified LDS CSV",
                csv_buffer.getvalue(),
                "lds_classified.csv",
                "text/csv"
            )
    with col2:
        st.download_button(
            "📸 Download Report PNG",
            data="Visualization ready - use browser screenshot tools",
            file_name="pilferage_report.png"
        )
    
else:
    st.info("👆 Please upload both PIDWS and LDS Excel files to start analysis")
