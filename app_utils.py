import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import tempfile

st.set_page_config(
    page_title="Pipeline Digging vs Leak Events",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛢️ Pipeline Digging vs Leak Events Analyzer")
st.markdown("Upload your PIDWS digging alarms and LDS leak data to visualize correlations by chainage with customizable tolerance.")

# Sidebar controls
st.sidebar.header("📊 Parameters")
tolerance = st.sidebar.slider("Chainage Tolerance (km)", 0.1, 10.0, 5.0, 0.1)
unique_chainages_option = st.sidebar.radio(
    "Chainage Selection",
    ["All unique chainages", "Specific chainage", "Top chainages by events"]
)
if unique_chainages_option == "Specific chainage":
    specific_chainage = st.sidebar.number_input("Enter chainage value", value=0.0)
if unique_chainages_option == "Top chainages by events":
    top_n = st.sidebar.slider("Top N chainages", 1, 20, 5)

# File uploaders
col1, col2 = st.columns(2)
with col1:
    digging_file = st.file_uploader("Upload Manual Digging Data (CSV/Excel)", type=['csv', 'xlsx'], key="digging")
with col2:
    leaks_file = st.file_uploader("Upload LDS Leak Data (CSV/Excel)", type=['csv', 'xlsx'], key="leaks")

if digging_file is not None and leaks_file is not None:
    # Load data
    @st.cache_data
    def load_data(digging_file, leaks_file):
        if digging_file.name.endswith('.csv'):
            df_manual_digging = pd.read_csv(digging_file)
        else:
            df_manual_digging = pd.read_excel(digging_file)
        
        if leaks_file.name.endswith('.csv'):
            df_lds_IV = pd.read_csv(leaks_file)
        else:
            df_lds_IV = pd.read_excel(leaks_file)
        
        # Ensure datetime columns exist and are parsed
        for df, dt_col in [(df_manual_digging, 'DateTime'), (df_lds_IV, 'DateTime')]:
            if dt_col not in df.columns:
                st.warning(f"No 'DateTime' column found in {dt_col}. Using index as fallback.")
            else:
                df[dt_col] = pd.to_datetime(df[dt_col])
        
        return df_manual_digging, df_lds_IV
    
    df_manual_digging, df_lds_IV = load_data(digging_file, leaks_file)
    
    st.success(f"✅ Loaded {len(df_manual_digging)} digging events and {len(df_lds_IV)} leak events.")
    
    # Data preview
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Digging Events Preview")
        st.dataframe(df_manual_digging[['DateTime', 'Original_chainage']].head())
    with col2:
        st.subheader("Leak Events Preview")
        st.dataframe(df_lds_IV[['DateTime', 'chainage']].head())
    
    # Get unique chainages
    digging_chainages = df_manual_digging['Original_chainage'].dropna().unique()
    leak_chainages = df_lds_IV['chainage'].dropna().unique()
    all_chainages = np.sort(np.unique(np.concatenate([digging_chainages, leak_chainages])))
    
    if unique_chainages_option == "Specific chainage":
        unique_chainages = [specific_chainage] if specific_chainage in all_chainages else []
    elif unique_chainages_option == "Top chainages by events":
        chainage_counts = {}
        for ch in all_chainages:
            dig_count = len(df_manual_digging[abs(df_manual_digging['Original_chainage'] - ch) <= tolerance])
            leak_count = len(df_lds_IV[abs(df_lds_IV['chainage'] - ch) <= tolerance])
            chainage_counts[ch] = dig_count + leak_count
        unique_chainages = sorted(chainage_counts, key=chainage_counts.get, reverse=True)[:top_n]
    else:
        unique_chainages = all_chainages
    
    st.subheader(f"📈 Plots for {len(unique_chainages)} Chainage(s)")
    
    # Plotting section
    for i, target_chainage_val in enumerate(unique_chainages):
        with st.container():
            df_digging_filtered = df_manual_digging[abs(df_manual_digging['Original_chainage'] - target_chainage_val) <= tolerance]
            df_leaks_filtered = df_lds_IV[abs(df_lds_IV['chainage'] - target_chainage_val) <= tolerance]
            
            if not df_digging_filtered.empty or not df_leaks_filtered.empty:
                fig = plt.figure(figsize=(18, 10))
                
                if not df_digging_filtered.empty:
                    plt.scatter(df_digging_filtered['DateTime'], df_digging_filtered['Original_chainage'], 
                               color='blue', label='Digging Events', marker='o', s=50)
                
                if not df_leaks_filtered.empty:
                    plt.scatter(df_leaks_filtered['DateTime'], df_leaks_filtered['chainage'], 
                               color='red', label='Leak Events', marker='X', s=80)
                
                plt.title(f'Digging vs. Leak Events at Chainage {target_chainage_val:.1f} (Tolerance: {tolerance:.1f} km)')
                plt.xlabel('Date and Time')
                plt.ylabel('Chainage (km)')
                plt.grid(True)
                plt.legend(title='Event Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close(fig)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Digging Events", len(df_digging_filtered))
                with col2:
                    st.metric("Leak Events", len(df_leaks_filtered))
                with col3:
                    st.metric("Total Events", len(df_digging_filtered) + len(df_leaks_filtered))
                
                st.markdown("---")
            else:
                st.info(f"No events found for chainage {target_chainage_val:.1f} with tolerance {tolerance:.1f} km.")
    
    # Download filtered data
    st.subheader("💾 Export Results")
    if st.button("Download All Filtered Data"):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for ch in unique_chainages[:10]:  # Limit to top 10 for performance
                df_dig = df_manual_digging[abs(df_manual_digging['Original_chainage'] - ch) <= tolerance]
                df_leak = df_lds_IV[abs(df_lds_IV['chainage'] - ch) <= tolerance]
                if not df_dig.empty:
                    df_dig.to_excel(writer, sheet_name=f'Digging_{ch:.1f}', index=False)
                if not df_leak.empty:
                    df_leak.to_excel(writer, sheet_name=f'Leaks_{ch:.1f}', index=False)
        st.download_button(
            "Download Excel",
            output.getvalue(),
            f"chainage_analysis_tolerance_{tolerance}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.warning("👆 Please upload both digging and leak data files to get started.")

st.markdown("---")
st.markdown("**Deploy Instructions:**")
st.markdown("- Save as `app.py` and `requirements.txt`")
st.markdown("- GitHub: Create repo → Add files → Go to Settings → Pages → Deploy from branch")
st.markdown("- Streamlit Cloud: Connect GitHub repo → Deploy instantly 🛠️")[web:15][web:21][memory:2]
