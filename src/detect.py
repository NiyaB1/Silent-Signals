"""
Main detection pipeline: load data, extract features, train model, detect anomalies, visualize.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import load_and_window_data
from feature_extraction import extract_features
from model import train_baseline_from_features, StressDetector


def run_pipeline(output_dir="results"):
    """Full pipeline: preprocess -> extract -> train -> detect -> visualize."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("SILENT SIGNALS - STRESS DETECTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and preprocess
    print("\n[1/5] Loading and preprocessing data...")
    try:
        windowed_data = load_and_window_data()
        print(f"✓ Loaded {len(windowed_data)} events")
        print(f"✓ Sessions: {windowed_data['session_id'].nunique()}")
        n_windows = windowed_data['window_id'].nunique()
        print(f"✓ Windows: {n_windows}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None
    
    # Step 2: Extract features
    print("\n[2/5] Extracting behavioral features...")
    try:
        features = extract_features()
        print(f"✓ Extracted {len(features)} window features")
        if features.empty:
            print("✗ No features extracted. Run logger first: python src/logger.py")
            return None
        print(f"✓ Feature columns: {', '.join([c for c in features.columns if c not in ['session_id', 'window_id']])}")
    except Exception as e:
        print(f"✗ Error extracting features: {e}")
        return None
    
    # Step 3: Train baseline model
    print("\n[3/5] Training baseline anomaly model...")
    try:
        baseline_model = train_baseline_from_features(features, contamination=0.05)
        print(f"✓ Baseline model trained on {len(features)} windows")
    except Exception as e:
        print(f"✗ Error training model: {e}")
        return None
    
    # Step 4: Detect anomalies
    print("\n[4/5] Detecting anomalies and computing stress risk...")
    try:
        detector = StressDetector(baseline_model)
        results = detector.evaluate(features)
        anomaly_count = results['is_anomaly'].sum()
        avg_risk = results['stress_risk'].mean()
        print(f"✓ Anomalies detected: {anomaly_count} / {len(results)}")
        print(f"✓ Average stress risk: {avg_risk:.3f}")
    except Exception as e:
        print(f"✗ Error in anomaly detection: {e}")
        return None
    
    # Step 5: Visualize and save
    print("\n[5/5] Generating visualizations...")
    try:
        visualize_results(results, output_dir)
        print(f"✓ Results saved to {output_dir}/")
    except Exception as e:
        print(f"✗ Error visualizing: {e}")
        return None
    
    print_summary(results)
    
    return {
        "windowed_data": windowed_data,
        "features": features,
        "results": results,
        "baseline_model": baseline_model,
        "detector": detector,
    }


def visualize_results(results, output_dir="results"):
    """Generate 4-panel visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Silent Signals - Stress Detection Results", fontsize=16, fontweight="bold")
    
    # Panel 1: Stress risk over time
    ax = axes[0, 0]
    for session_id in results['session_id'].unique():
        session_data = results[results['session_id'] == session_id].copy()
        session_data = session_data.sort_values('window_id')
        ax.plot(session_data['window_id'], session_data['stress_risk'],
                marker='o', label=f"Session {session_id[:8]}", alpha=0.7, linewidth=2)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate')
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High')
    ax.set_xlabel('Window ID', fontsize=10)
    ax.set_ylabel('Stress Risk (0-1)', fontsize=10)
    ax.set_title('Stress Risk Timeline', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Anomaly count per session
    ax = axes[0, 1]
    anomaly_by_session = results.groupby('session_id')['is_anomaly'].sum()
    anomaly_by_session.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
    ax.set_xlabel('Session', fontsize=10)
    ax.set_ylabel('Anomaly Count', fontsize=10)
    ax.set_title('Anomalies per Session', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Feature statistics boxplot
    ax = axes[1, 0]
    feature_cols = ['key_presses', 'mouse_activity', 'typing_speed']
    feature_subset = results[feature_cols]
    feature_subset.boxplot(ax=ax)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Feature Distributions', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Stress risk histogram
    ax = axes[1, 1]
    ax.hist(results['stress_risk'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    mean_risk = results['stress_risk'].mean()
    ax.axvline(x=mean_risk, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_risk:.3f}')
    ax.set_xlabel('Stress Risk', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Stress Risk Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, "stress_report.png")
    plt.savefig(plot_file, dpi=100, bbox_inches='tight')
    print(f"  → Saved: {plot_file}")
    plt.close()
    
    # Save results CSV
    csv_file = os.path.join(output_dir, "detection_results.csv")
    results.to_csv(csv_file, index=False)
    print(f"  → Saved: {csv_file}")


def print_summary(results):
    """Print summary statistics."""
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Per-session stats
    by_session = results.groupby('session_id').agg({
        'stress_risk': ['mean', 'max', 'std'],
        'is_anomaly': 'sum',
        'window_id': 'count'
    }).round(3)
    
    print("\nPer-Session Stress Report:")
    print(by_session)
    
    # Overall stats
    overall_mean = results['stress_risk'].mean()
    overall_max = results['stress_risk'].max()
    high_risk_windows = (results['stress_risk'] > 0.7).sum()
    total_windows = len(results)
    
    print(f"\nOverall Statistics:")
    print(f"  Average stress risk: {overall_mean:.3f}")
    print(f"  Max stress risk: {overall_max:.3f}")
    print(f"  High-risk windows (>0.7): {high_risk_windows} / {total_windows}")
    
    # Interpretation
    print("\nInterpretation:")
    if overall_mean < 0.3:
        print("  ✓ Stress levels NORMAL")
    elif overall_mean < 0.5:
        print("  ⚡ Stress levels ELEVATED")
    else:
        print("  ⚠️  WARNING: HIGH stress detected")


if __name__ == "__main__":
    run_pipeline()
