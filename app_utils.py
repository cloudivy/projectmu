
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Parsing helpers
# -----------------------------
def parse_duration(dur_str):
    """Parse duration strings like '10m 30s', '5m', '45s' into Timedelta."""
    if pd.isna(dur_str):
        return pd.Timedelta(0)
    s = str(dur_str).strip().lower().replace(' ', '')
    mins = 0
    secs = 0
    if 'm' in s:
        m_part = s.split('m')[0]
        if m_part.isdigit():
            mins = int(m_part)
        s = s.split('m')[1]
    if 's' in s:
        s_part = s.replace('s', '')
        if s_part.isdigit():
            secs = int(s_part)
    return pd.Timedelta(minutes=mins, seconds=secs)

def try_parse_datetime(date_str, time_str):
    """Attempt robust parsing of date+time."""
    combo = f"{date_str} {time_str}"
    fmts = [
        '%d-%m-%Y %H:%M:%S',
        '%d/%m/%Y %H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
        '%d-%m-%Y %H:%M',
        '%Y-%m-%d %H:%M',
    ]
    for fmt in fmts:
        try:
            return pd.to_datetime(combo, format=fmt, dayfirst=True, errors='raise')
        except Exception:
            continue
    return pd.to_datetime(combo, dayfirst=True, errors='coerce')

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_pidws(df_pidws: pd.DataFrame) -> pd.DataFrame:
    df = df_pidws.copy()
    df['DateTime'] = [
        try_parse_datetime(d, t) for d, t in zip(df['Date'].astype(str), df['Time'].astype(str))
    ]
    df['duration_td'] = df['Event Duration'].apply(parse_duration)
    df['end_time'] = df['DateTime'] + df['duration_td']
    return df

def preprocess_lds(df_lds: pd.DataFrame) -> pd.DataFrame:
    df = df_lds.copy()
    df['DateTime'] = [
        try_parse_datetime(d, t) for d, t in zip(df['Date'].astype(str), df['Time'].astype(str))
    ]
    return df

# -----------------------------
# Classification
# -----------------------------
def classify_pilferage(pidws_df: pd.DataFrame, lds_df: pd.DataFrame,
                       chainage_tol: float = 1.0, time_window_hours: int = 24) -> pd.DataFrame:
    """
    Classify LDS leaks as pilferage if they occur after PIDWS end_time + window
    and are within chainage tolerance.
    Returns a DataFrame of matched LDS rows with metadata and score.
    """
    classified = []
    for _, event in pidws_df.iterrows():
        window_end = event['end_time'] + pd.Timedelta(hours=time_window_hours)
        mask = (lds_df['DateTime'] > window_end) & (np.abs(lds_df['chainage'] - event['chainage']) <= chainage_tol)
        matches = lds_df.loc[mask].copy()
        if not matches.empty:
            matches['linked_event_time'] = event['DateTime']
            matches['linked_chainage'] = event['chainage']
            matches['pilferage_score'] = 1.0 / (1.0 + (matches['DateTime'] - window_end).dt.total_seconds() / 3600.0)
            classified.append(matches)
    if classified:
        return pd.concat(classified, ignore_index=True)
    return pd.DataFrame()

def flag_pilferage_in_lds(lds_df: pd.DataFrame, pilferage_df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flag `is_pilferage` to lds_df based on DateTime+chainage matches."""
    out = lds_df.copy()
    out['is_pilferage'] = False
    if pilferage_df is not None and not pilferage_df.empty:
        ids = pilferage_df[['DateTime', 'chainage']].drop_duplicates()
        mask = out.set_index(['DateTime', 'chainage']).index.isin(ids.set_index(['DateTime', 'chainage']).index)
        out.loc[mask, 'is_pilferage'] = True
    return out

# -----------------------------
# Visualization
# -----------------------------
def build_visualizations(pidws_df: pd.DataFrame, lds_df: pd.DataFrame, pilferage_df: pd.DataFrame):
    """Create a 2x2 matplotlib figure with four panels as per spec."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Chainage distribution
    axes[0, 0].hist(pidws_df['chainage'], bins=30, alpha=0.7, label='PIDWS Digging', color='orange')
    axes[0, 0].hist(lds_df['chainage'], bins=30, alpha=0.7, label='LDS Leaks', color='blue')
    pilf_mean = pilferage_df['linked_chainage'].mean() if pilferage_df is not None and not pilferage_df.empty else None
    if pilf_mean is not None and pd.notna(pilf_mean):
        axes[0, 0].axvline(pilf_mean, color='red', linestyle='--', label='Pilferage Mean')
    axes[0, 0].set_xlabel('Chainage (km)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].set_title('Chainage Distribution')

    # 2. Temporal Pattern (events per DateTime)
    all_events = pd.concat([
        pidws_df[['DateTime', 'chainage']].assign(type='Digging'),
        lds_df[['DateTime', 'chainage']].assign(type='Leak'),
        (pilferage_df[['DateTime', 'linked_chainage']].rename(columns={'linked_chainage': 'chainage'}).assign(type='Pilferage')
         if pilferage_df is not None and not pilferage_df.empty else pd.DataFrame(columns=['DateTime', 'chainage', 'type']))
    ], ignore_index=True)
    if not all_events.empty:
        time_pivot = all_events.groupby(['DateTime', 'type']).size().unstack(fill_value=0)
        time_pivot.plot(ax=axes[0, 1], linewidth=1)
        axes[0, 1].set_title('Temporal Pattern')
        axes[0, 1].set_ylabel('Events per Timestamp')
    else:
        axes[0, 1].text(0.5, 0.5, 'No events to plot', ha='center')
        axes[0, 1].set_axis_off()

    # 3. Leak size vs classification
    lds_classified_temp = flag_pilferage_in_lds(lds_df, pilferage_df)
    if 'leak size' in lds_classified_temp.columns and not lds_classified_temp.empty:
        sns.boxplot(data=lds_classified_temp, x='is_pilferage', y='leak size', ax=axes[1, 0])
        axes[1, 0].set_title('Leak Size: Pilferage vs Others')
    else:
        axes[1, 0].text(0.5, 0.5, 'No leak size data', ha='center')
        axes[1, 0].set_axis_off()

    # 4. Chainage scatter with classification
    colors = ['red' if x else 'blue' for x in lds_classified_temp['is_pilferage']] if not lds_classified_temp.empty else []
    if colors:
        axes[1, 1].scatter(lds_classified_temp['DateTime'], lds_classified_temp['chainage'], c=colors, alpha=0.6, s=20)
        axes[1, 1].set_xlabel('DateTime')
        axes[1, 1].set_ylabel('Chainage')
        axes[1, 1].set_title('Leaks: Red=Pilferage, Blue=Other')
    else:
        axes[1, 1].text(0.5, 0.5, 'No LDS rows', ha='center')
        axes[1, 1].set_axis_off()

    plt.tight_layout()
    return fig
