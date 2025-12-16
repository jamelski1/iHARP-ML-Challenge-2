#!/usr/bin/env python3
"""
Station Trend Analysis & Anomaly Detection
==========================================
Visualize coastal flooding trends for each station to identify:
- Anomalies and outliers
- Sensor issues (gaps, drift, sudden jumps)
- Long-term trends and seasonality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.io import loadmat
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = 'NEUSTG_19502020_12stations.mat'
OUTPUT_DIR = 'station_analysis_plots'

def matlab2datetime(matlab_datenum):
    """Convert MATLAB datenum to Python datetime."""
    return datetime.fromordinal(int(matlab_datenum)) + \
           timedelta(days=matlab_datenum % 1) - timedelta(days=366)

def load_station_data(filepath=DATA_FILE):
    """Load and prepare station data from .mat file."""
    print("Loading data from", filepath)
    mat_data = loadmat(filepath)

    # Extract arrays
    sea_level = mat_data['sltg']  # (time, stations)
    latitudes = mat_data['lattg'].flatten()
    longitudes = mat_data['lontg'].flatten()
    time_array = mat_data['t'].flatten()
    station_names = [str(s).strip() for s in mat_data['sname'].flatten()]

    # Clean station names
    station_names = [s.replace("'", "").replace("[", "").replace("]", "").strip()
                     for s in station_names]

    # Convert time
    print("Converting timestamps...")
    timestamps = [matlab2datetime(t) for t in time_array]

    # Build DataFrame
    print("Building DataFrame...")
    records = []
    for i, station in enumerate(station_names):
        for j, ts in enumerate(timestamps):
            records.append({
                'time': ts,
                'station': station,
                'sea_level': sea_level[j, i],
                'latitude': latitudes[i],
                'longitude': longitudes[i]
            })

    df = pd.DataFrame(records)
    df['time'] = pd.to_datetime(df['time'])

    print(f"Loaded {len(df):,} records from {len(station_names)} stations")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")

    return df, station_names

def detect_anomalies(series, window=24*7, threshold=3.5):
    """
    Detect anomalies using modified Z-score with rolling statistics.
    Returns boolean mask of anomalies.
    """
    rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
    rolling_mad = series.rolling(window=window, center=True, min_periods=1).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )

    # Modified Z-score
    modified_z = 0.6745 * (series - rolling_median) / (rolling_mad + 1e-10)

    return np.abs(modified_z) > threshold

def detect_sensor_issues(series, timestamps):
    """
    Detect potential sensor issues:
    - Data gaps
    - Sudden jumps (step changes)
    - Stuck values (no variation)
    - Drift
    """
    issues = {
        'gaps': [],
        'jumps': [],
        'stuck': [],
        'drift_periods': []
    }

    # Detect gaps (missing data or NaN sequences)
    nan_mask = series.isna()
    if nan_mask.any():
        # Find contiguous NaN regions
        nan_diff = nan_mask.astype(int).diff()
        gap_starts = timestamps[nan_diff == 1].tolist()
        gap_ends = timestamps[nan_diff == -1].tolist()
        issues['gaps'] = list(zip(gap_starts[:len(gap_ends)], gap_ends))

    # Detect sudden jumps (step changes)
    diff = series.diff()
    jump_threshold = diff.std() * 5
    jumps = np.abs(diff) > jump_threshold
    if jumps.any():
        issues['jumps'] = timestamps[jumps].tolist()

    # Detect stuck sensor (no variation over extended period)
    rolling_std = series.rolling(window=24*3, min_periods=12).std()
    stuck = rolling_std < 0.001  # Nearly zero variation for 3 days
    if stuck.any():
        issues['stuck'] = timestamps[stuck].tolist()

    return issues

def compute_flood_threshold(series, method='percentile', percentile=95):
    """Compute flood threshold for a station."""
    clean_series = series.dropna()
    if method == 'percentile':
        return np.percentile(clean_series, percentile)
    elif method == 'std':
        return clean_series.mean() + 1.5 * clean_series.std()
    else:
        return np.percentile(clean_series, 95)

def plot_station_overview(df, station, save_dir=OUTPUT_DIR):
    """Create comprehensive overview plot for a single station."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    station_df = df[df['station'] == station].copy()
    station_df = station_df.sort_values('time').reset_index(drop=True)

    # Resample to daily for cleaner visualization
    daily = station_df.set_index('time').resample('D').agg({
        'sea_level': ['mean', 'max', 'min', 'std']
    }).dropna()
    daily.columns = ['mean', 'max', 'min', 'std']
    daily = daily.reset_index()

    # Compute threshold and anomalies
    threshold = compute_flood_threshold(daily['max'])
    anomalies = detect_anomalies(daily['mean'])
    flood_days = daily['max'] > threshold

    # Create figure
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[2, 1, 1, 1])

    # =========================================================================
    # Plot 1: Full time series with threshold and anomalies
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :])

    # Plot daily mean with range
    ax1.fill_between(daily['time'], daily['min'], daily['max'],
                     alpha=0.3, color='steelblue', label='Daily range')
    ax1.plot(daily['time'], daily['mean'], 'b-', linewidth=0.5,
             alpha=0.7, label='Daily mean')

    # Highlight anomalies
    anomaly_times = daily.loc[anomalies, 'time']
    anomaly_values = daily.loc[anomalies, 'mean']
    ax1.scatter(anomaly_times, anomaly_values, c='red', s=20,
                zorder=5, label=f'Anomalies ({anomalies.sum()})', alpha=0.7)

    # Flood threshold line
    ax1.axhline(y=threshold, color='orange', linestyle='--',
                linewidth=2, label=f'Flood threshold ({threshold:.2f}m)')

    ax1.set_title(f'{station} - Sea Level Time Series (1950-2020)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Sea Level (m)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 2: Annual flood frequency trend
    # =========================================================================
    ax2 = fig.add_subplot(gs[1, 0])

    daily['year'] = daily['time'].dt.year
    annual_floods = daily.groupby('year')['max'].apply(
        lambda x: (x > threshold).sum()
    ).reset_index()
    annual_floods.columns = ['year', 'flood_days']

    # Trend line
    z = np.polyfit(annual_floods['year'], annual_floods['flood_days'], 1)
    p = np.poly1d(z)
    trend_direction = "↑ Increasing" if z[0] > 0 else "↓ Decreasing"

    ax2.bar(annual_floods['year'], annual_floods['flood_days'],
            color='coral', alpha=0.7, edgecolor='darkred')
    ax2.plot(annual_floods['year'], p(annual_floods['year']),
             'b--', linewidth=2, label=f'Trend: {trend_direction}')

    ax2.set_title('Annual Flood Days (above threshold)', fontsize=12)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Days above threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 3: Monthly seasonality
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 1])

    daily['month'] = daily['time'].dt.month
    monthly_stats = daily.groupby('month')['mean'].agg(['mean', 'std']).reset_index()

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    ax3.bar(monthly_stats['month'], monthly_stats['mean'],
            yerr=monthly_stats['std'], capsize=3,
            color='teal', alpha=0.7, edgecolor='darkslategray')
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(months, rotation=45)
    ax3.set_title('Monthly Sea Level Pattern', fontsize=12)
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Mean Sea Level (m)')
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 4: Distribution and outliers
    # =========================================================================
    ax4 = fig.add_subplot(gs[2, 0])

    # Histogram with outlier markers
    clean_data = daily['mean'].dropna()
    ax4.hist(clean_data, bins=100, density=True, alpha=0.7,
             color='steelblue', edgecolor='navy')

    # Mark percentiles
    p5, p95 = np.percentile(clean_data, [5, 95])
    ax4.axvline(p5, color='orange', linestyle='--', label=f'5th %ile: {p5:.2f}m')
    ax4.axvline(p95, color='red', linestyle='--', label=f'95th %ile: {p95:.2f}m')
    ax4.axvline(threshold, color='darkred', linestyle='-',
                linewidth=2, label=f'Flood threshold: {threshold:.2f}m')

    ax4.set_title('Sea Level Distribution', fontsize=12)
    ax4.set_xlabel('Sea Level (m)')
    ax4.set_ylabel('Density')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 5: Box plot by decade
    # =========================================================================
    ax5 = fig.add_subplot(gs[2, 1])

    daily['decade'] = (daily['year'] // 10) * 10
    decades = sorted(daily['decade'].unique())
    decade_data = [daily[daily['decade'] == d]['mean'].dropna().values for d in decades]

    bp = ax5.boxplot(decade_data, labels=[f"{d}s" for d in decades],
                     patch_artist=True)

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(decades)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax5.set_title('Sea Level by Decade', fontsize=12)
    ax5.set_xlabel('Decade')
    ax5.set_ylabel('Sea Level (m)')
    ax5.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 6: Rate of change analysis
    # =========================================================================
    ax6 = fig.add_subplot(gs[3, 0])

    # 30-day rate of change
    daily['rate_30d'] = daily['mean'].diff(30) / 30  # m per day

    ax6.plot(daily['time'], daily['rate_30d'], 'g-', alpha=0.5, linewidth=0.5)

    # Highlight extreme rates
    extreme_rate = daily['rate_30d'].abs() > daily['rate_30d'].std() * 3
    ax6.scatter(daily.loc[extreme_rate, 'time'],
                daily.loc[extreme_rate, 'rate_30d'],
                c='red', s=10, alpha=0.5, label='Extreme rate changes')

    ax6.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax6.set_title('30-Day Rate of Change', fontsize=12)
    ax6.set_xlabel('Year')
    ax6.set_ylabel('Rate (m/day)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 7: Data quality / sensor issues
    # =========================================================================
    ax7 = fig.add_subplot(gs[3, 1])

    # Missing data by year
    hourly_station = station_df.set_index('time')
    yearly_missing = hourly_station.resample('Y')['sea_level'].apply(
        lambda x: x.isna().sum() / len(x) * 100 if len(x) > 0 else 0
    )

    years = yearly_missing.index.year
    ax7.bar(years, yearly_missing.values, color='crimson', alpha=0.7)
    ax7.set_title('Data Quality: Missing Data %', fontsize=12)
    ax7.set_xlabel('Year')
    ax7.set_ylabel('% Missing')
    ax7.grid(True, alpha=0.3)

    # Add alert if high missing data
    if yearly_missing.max() > 10:
        ax7.annotate(f'⚠ Max missing: {yearly_missing.max():.1f}%',
                     xy=(0.5, 0.95), xycoords='axes fraction',
                     fontsize=10, color='red', ha='center')

    plt.tight_layout()

    # Save figure
    filepath = f"{save_dir}/{station}_analysis.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()

    return {
        'station': station,
        'threshold': threshold,
        'total_anomalies': anomalies.sum(),
        'total_flood_days': flood_days.sum(),
        'trend_slope': z[0],
        'max_missing_pct': yearly_missing.max()
    }

def plot_all_stations_comparison(df, station_names, save_dir=OUTPUT_DIR):
    """Create comparison plots across all stations."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.flatten()

    station_stats = []

    for idx, station in enumerate(station_names):
        ax = axes[idx]
        station_df = df[df['station'] == station].copy()

        # Resample to monthly for comparison
        monthly = station_df.set_index('time').resample('M').agg({
            'sea_level': 'mean'
        }).dropna()

        # Compute threshold
        threshold = compute_flood_threshold(monthly['sea_level'])

        # Plot
        ax.plot(monthly.index, monthly['sea_level'], 'b-', linewidth=0.5, alpha=0.7)
        ax.axhline(threshold, color='red', linestyle='--', alpha=0.7)

        # 5-year rolling average for trend
        rolling_5y = monthly['sea_level'].rolling(window=60, min_periods=30).mean()
        ax.plot(monthly.index, rolling_5y, 'orange', linewidth=2, label='5-yr avg')

        ax.set_title(f'{station}', fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        # Collect stats
        lat = station_df['latitude'].iloc[0]
        lon = station_df['longitude'].iloc[0]
        station_stats.append({
            'station': station,
            'latitude': lat,
            'longitude': lon,
            'mean_level': monthly['sea_level'].mean(),
            'std_level': monthly['sea_level'].std(),
            'threshold': threshold
        })

    plt.suptitle('Sea Level Trends: All 12 Stations (Monthly Averages)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = f"{save_dir}/all_stations_comparison.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

    return pd.DataFrame(station_stats)

def plot_anomaly_summary(df, station_names, save_dir=OUTPUT_DIR):
    """Create summary of anomalies across all stations."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    anomaly_counts = []

    for station in station_names:
        station_df = df[df['station'] == station].copy()
        daily = station_df.set_index('time').resample('D')['sea_level'].mean().dropna()

        anomalies = detect_anomalies(daily)
        issues = detect_sensor_issues(daily, daily.index.to_series())

        anomaly_counts.append({
            'station': station,
            'anomaly_count': anomalies.sum(),
            'anomaly_pct': anomalies.sum() / len(daily) * 100,
            'jump_count': len(issues['jumps']),
            'gap_count': len(issues['gaps'])
        })

    anomaly_df = pd.DataFrame(anomaly_counts)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Anomaly counts
    ax1 = axes[0]
    bars = ax1.barh(anomaly_df['station'], anomaly_df['anomaly_count'], color='coral')
    ax1.set_xlabel('Number of Anomalies')
    ax1.set_title('Detected Anomalies by Station', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Color code by severity
    for bar, count in zip(bars, anomaly_df['anomaly_count']):
        if count > anomaly_df['anomaly_count'].median() * 2:
            bar.set_color('darkred')

    # Plot 2: Jump counts (sensor issues)
    ax2 = axes[1]
    ax2.barh(anomaly_df['station'], anomaly_df['jump_count'], color='purple', alpha=0.7)
    ax2.set_xlabel('Number of Sudden Jumps')
    ax2.set_title('Sensor Jump Events by Station', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # Plot 3: Anomaly percentage
    ax3 = axes[2]
    ax3.barh(anomaly_df['station'], anomaly_df['anomaly_pct'], color='teal', alpha=0.7)
    ax3.set_xlabel('Anomaly %')
    ax3.set_title('Anomaly Rate by Station', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    filepath = f"{save_dir}/anomaly_summary.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

    return anomaly_df

def generate_text_report(station_results, anomaly_df):
    """Generate a text summary report."""
    report = []
    report.append("=" * 70)
    report.append("COASTAL FLOODING TREND ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")

    # Overall summary
    report.append("SUMMARY")
    report.append("-" * 40)
    total_anomalies = sum(r['total_anomalies'] for r in station_results)
    report.append(f"Total stations analyzed: {len(station_results)}")
    report.append(f"Total anomalies detected: {total_anomalies}")
    report.append("")

    # Stations with concerning trends
    report.append("STATIONS WITH INCREASING FLOOD TRENDS")
    report.append("-" * 40)
    increasing = [r for r in station_results if r['trend_slope'] > 0]
    for r in sorted(increasing, key=lambda x: -x['trend_slope']):
        report.append(f"  {r['station']}: +{r['trend_slope']:.3f} flood days/year")
    report.append("")

    # Data quality issues
    report.append("DATA QUALITY CONCERNS")
    report.append("-" * 40)
    for r in station_results:
        if r['max_missing_pct'] > 5:
            report.append(f"  ⚠ {r['station']}: {r['max_missing_pct']:.1f}% missing data in worst year")
    report.append("")

    # High anomaly stations
    report.append("HIGH ANOMALY STATIONS (> 2x median)")
    report.append("-" * 40)
    median_anomalies = anomaly_df['anomaly_count'].median()
    for _, row in anomaly_df.iterrows():
        if row['anomaly_count'] > median_anomalies * 2:
            report.append(f"  ⚠ {row['station']}: {row['anomaly_count']} anomalies ({row['anomaly_pct']:.2f}%)")

    report.append("")
    report.append("=" * 70)

    return "\n".join(report)

def main():
    """Main execution function."""
    import os

    print("=" * 60)
    print("STATION TREND ANALYSIS & ANOMALY DETECTION")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df, station_names = load_station_data()

    print("\n" + "=" * 60)
    print("Generating station-level analysis plots...")
    print("=" * 60)

    # Individual station analysis
    station_results = []
    for station in station_names:
        print(f"\nAnalyzing: {station}")
        result = plot_station_overview(df, station)
        station_results.append(result)

    print("\n" + "=" * 60)
    print("Generating comparison plots...")
    print("=" * 60)

    # Cross-station comparisons
    station_stats = plot_all_stations_comparison(df, station_names)
    print("\nStation Statistics:")
    print(station_stats.to_string(index=False))

    # Anomaly summary
    print("\n" + "=" * 60)
    print("Generating anomaly summary...")
    print("=" * 60)
    anomaly_df = plot_anomaly_summary(df, station_names)
    print("\nAnomaly Summary:")
    print(anomaly_df.to_string(index=False))

    # Generate text report
    report = generate_text_report(station_results, anomaly_df)
    print("\n" + report)

    # Save report
    report_path = f"{OUTPUT_DIR}/analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    print("\n" + "=" * 60)
    print(f"All plots saved to: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
