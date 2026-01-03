#!/usr/bin/env python3
"""
Analyze per-video TTA results to understand which video categories benefit most.

This script analyzes the detailed_metrics.csv from evaluation to identify:
1. Which UCF-101 action categories benefit most from TTA
2. Patterns in improvement (motion type, complexity, etc.)
3. Percentage of videos showing improvement vs degradation

Usage:
    python analyze_per_video.py --metrics-file /path/to/detailed_metrics.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def extract_class_from_video_name(video_name: str) -> str:
    """Extract class name from video filename.
    
    Format: v_ClassName_g##_c## -> ClassName
    """
    parts = video_name.split('_')
    if len(parts) >= 2:
        return parts[1]
    return video_name


def analyze_metrics(df: pd.DataFrame) -> dict:
    """Analyze per-video metrics."""
    
    results = {}
    
    # Extract class from video name
    df['class'] = df['video_name'].apply(extract_class_from_video_name)
    
    # Overall statistics
    results['total_videos'] = len(df)
    
    # PSNR improvement analysis
    if 'psnr_improvement' in df.columns:
        psnr_imp = df['psnr_improvement'].dropna()
        results['psnr'] = {
            'mean': float(psnr_imp.mean()),
            'std': float(psnr_imp.std()),
            'median': float(psnr_imp.median()),
            'improved': int((psnr_imp > 0).sum()),
            'degraded': int((psnr_imp < 0).sum()),
            'unchanged': int((psnr_imp == 0).sum()),
            'improvement_rate': float((psnr_imp > 0).mean() * 100),
        }
    
    # SSIM improvement analysis
    if 'ssim_improvement' in df.columns:
        ssim_imp = df['ssim_improvement'].dropna()
        results['ssim'] = {
            'mean': float(ssim_imp.mean()),
            'std': float(ssim_imp.std()),
            'improved': int((ssim_imp > 0).sum()),
            'degraded': int((ssim_imp < 0).sum()),
            'improvement_rate': float((ssim_imp > 0).mean() * 100),
        }
    
    # LPIPS improvement analysis
    if 'lpips_improvement' in df.columns:
        lpips_imp = df['lpips_improvement'].dropna()
        results['lpips'] = {
            'mean': float(lpips_imp.mean()),
            'std': float(lpips_imp.std()),
            'improved': int((lpips_imp > 0).sum()),
            'degraded': int((lpips_imp < 0).sum()),
            'improvement_rate': float((lpips_imp > 0).mean() * 100),
        }
    
    # Per-class analysis
    if 'psnr_improvement' in df.columns:
        class_stats = df.groupby('class').agg({
            'psnr_improvement': ['mean', 'std', 'count'],
        }).round(4)
        
        # Flatten column names
        class_stats.columns = ['psnr_mean', 'psnr_std', 'count']
        class_stats = class_stats.sort_values('psnr_mean', ascending=False)
        
        results['top_classes'] = class_stats.head(10).to_dict('index')
        results['bottom_classes'] = class_stats.tail(10).to_dict('index')
    
    return results


def print_analysis(results: dict):
    """Print formatted analysis results."""
    
    print("=" * 70)
    print("Per-Video TTA Analysis")
    print("=" * 70)
    
    print(f"\nTotal videos analyzed: {results['total_videos']}")
    
    # PSNR Analysis
    if 'psnr' in results:
        psnr = results['psnr']
        print(f"\n--- PSNR Improvement ---")
        print(f"  Mean: {psnr['mean']:+.4f}")
        print(f"  Std:  {psnr['std']:.4f}")
        print(f"  Median: {psnr['median']:+.4f}")
        print(f"  Improved: {psnr['improved']} videos ({psnr['improvement_rate']:.1f}%)")
        print(f"  Degraded: {psnr['degraded']} videos ({100 - psnr['improvement_rate']:.1f}%)")
    
    # SSIM Analysis
    if 'ssim' in results:
        ssim = results['ssim']
        print(f"\n--- SSIM Improvement ---")
        print(f"  Mean: {ssim['mean']:+.6f}")
        print(f"  Improved: {ssim['improved']} videos ({ssim['improvement_rate']:.1f}%)")
        print(f"  Degraded: {ssim['degraded']} videos ({100 - ssim['improvement_rate']:.1f}%)")
    
    # LPIPS Analysis
    if 'lpips' in results:
        lpips = results['lpips']
        print(f"\n--- LPIPS Improvement ---")
        print(f"  Mean: {lpips['mean']:+.6f}")
        print(f"  Improved: {lpips['improved']} videos ({lpips['improvement_rate']:.1f}%)")
        print(f"  Degraded: {lpips['degraded']} videos ({100 - lpips['improvement_rate']:.1f}%)")
    
    # Top classes
    if 'top_classes' in results:
        print(f"\n--- Top 10 Classes (Most Benefit from TTA) ---")
        for i, (cls, stats) in enumerate(results['top_classes'].items(), 1):
            print(f"  {i:2d}. {cls}: PSNR {stats['psnr_mean']:+.4f} ± {stats['psnr_std']:.4f} (n={stats['count']})")
    
    # Bottom classes
    if 'bottom_classes' in results:
        print(f"\n--- Bottom 10 Classes (Least Benefit or Hurt by TTA) ---")
        for i, (cls, stats) in enumerate(results['bottom_classes'].items(), 1):
            print(f"  {i:2d}. {cls}: PSNR {stats['psnr_mean']:+.4f} ± {stats['psnr_std']:.4f} (n={stats['count']})")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze per-video TTA results")
    parser.add_argument(
        "--metrics-file",
        type=str,
        required=True,
        help="Path to detailed_metrics.csv from evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Save analysis to JSON file"
    )
    args = parser.parse_args()
    
    # Load metrics
    metrics_path = Path(args.metrics_file)
    if not metrics_path.exists():
        print(f"Error: Metrics file not found at {metrics_path}")
        return
    
    df = pd.read_csv(metrics_path)
    print(f"Loaded {len(df)} video results from {metrics_path}")
    
    # Analyze
    results = analyze_metrics(df)
    
    # Print
    print_analysis(results)
    
    # Save if requested
    if args.output:
        import json
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nAnalysis saved to {output_path}")


if __name__ == "__main__":
    main()

