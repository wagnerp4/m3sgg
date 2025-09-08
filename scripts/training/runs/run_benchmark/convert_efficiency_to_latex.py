#!/usr/bin/env python3
"""
Convert the computational efficiency table to LaTeX format.
"""

import pandas as pd
import os

def create_efficiency_latex_table():
    """Create LaTeX table from the computational efficiency CSV data."""
    
    # Read the CSV file
    csv_file = 'data/benchmark/ag200/computational_efficiency_table.csv'
    df = pd.read_csv(csv_file)
    
    # Create LaTeX table
    latex_code = """
\\begin{table}[htbp]
\\centering
\\caption{Computational Efficiency Analysis for All Models and Modes on AG200 Dataset}
\\label{tab:computational_efficiency}
\\begin{tabular}{lcccccccc}
\\toprule
\\textbf{Model} & \\textbf{Mode} & \\textbf{Duration (min)} & \\textbf{Epochs} & \\textbf{Avg Epoch (min)} & \\textbf{GPU Util (\\%)} & \\textbf{Throughput (s/s)} & \\textbf{R@20} & \\textbf{mR@20} \\\\
\\midrule
"""
    
    # Add data rows
    for _, row in df.iterrows():
        model = row['Model']
        mode = row['Mode']
        duration = row['Duration (min)']
        epochs = row['Epochs']
        avg_epoch = row['Avg Epoch Time (min)']
        gpu_util = row['GPU Utilization (%)']
        throughput = row['Throughput (samples/s)']
        r20 = row['Final R@20']
        mr20 = row['Final mR@20']
        
        latex_code += f"{model} & {mode} & {duration} & {epochs} & {avg_epoch} & {gpu_util} & {throughput} & {r20} & {mr20} \\\\\n"
    
    # Close the table
    latex_code += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Training was performed for 10 epochs with 20\% of the AG200 dataset (fraction=5). Duration includes both training and evaluation time. GPU utilization is measured at the end of training. Throughput is calculated as training samples per second. Note that STTRAN and DSG-DETR show zero performance on sgcls mode, indicating potential training issues.
\\end{tablenotes}
\\end{table}
"""
    
    return latex_code

def create_efficiency_summary_latex_table():
    """Create a summary LaTeX table with key efficiency metrics."""
    
    # Read the summary CSV file
    csv_file = 'data/benchmark/ag200/computational_efficiency_summary.csv'
    df = pd.read_csv(csv_file)
    
    # Create summary table
    latex_code = """
\\begin{table}[htbp]
\\centering
\\caption{Computational Efficiency Summary - Average Performance Across All Modes}
\\label{tab:efficiency_summary}
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Model} & \\textbf{Avg Duration (min)} & \\textbf{Avg Epoch (min)} & \\textbf{Avg GPU Util (\\%)} & \\textbf{Avg Throughput (s/s)} & \\textbf{Avg R@20} & \\textbf{Avg mR@20} \\\\
\\midrule
"""
    
    # Add data rows
    for _, row in df.iterrows():
        model = row['Model']
        avg_duration = row['Avg Duration (min)']
        avg_epoch = row['Avg Epoch Time (min)']
        avg_gpu_util = row['Avg GPU Util (%)']
        avg_throughput = row['Avg Throughput (samples/s)']
        avg_r20 = row['Avg R@20']
        avg_mr20 = row['Avg mR@20']
        
        latex_code += f"{model} & {avg_duration} & {avg_epoch} & {avg_gpu_util} & {avg_throughput} & {avg_r20} & {avg_mr20} \\\\\n"
    
    # Close the table
    latex_code += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item TEMPURA shows the best average performance (mR@20: 0.1069) but requires the longest training time. STTRAN is the fastest overall but has lower performance. DSG-DETR provides a good balance between speed and performance.
\\end{tablenotes}
\\end{table}
"""
    
    return latex_code

def create_performance_efficiency_latex_table():
    """Create a table focusing on performance vs efficiency trade-offs."""
    
    # Read the CSV file
    csv_file = 'data/benchmark/ag200/computational_efficiency_table.csv'
    df = pd.read_csv(csv_file)
    
    # Filter out failed runs (sgcls with 0 performance)
    df_filtered = df[df['Final mR@20'] != '0.0000']
    
    # Create performance-efficiency table
    latex_code = """
\\begin{table}[htbp]
\\centering
\\caption{Performance vs Efficiency Trade-offs (Successful Runs Only)}
\\label{tab:performance_efficiency}
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Model} & \\textbf{Mode} & \\textbf{Duration (min)} & \\textbf{Throughput (s/s)} & \\textbf{R@20} & \\textbf{mR@20} & \\textbf{Efficiency Score} \\\\
\\midrule
"""
    
    # Add data rows with efficiency score calculation
    for _, row in df_filtered.iterrows():
        model = row['Model']
        mode = row['Mode']
        duration = row['Duration (min)']
        throughput = row['Throughput (samples/s)']
        r20 = row['Final R@20']
        mr20 = row['Final mR@20']
        
        # Calculate efficiency score (mR@20 per minute of training)
        try:
            efficiency_score = float(mr20) / float(duration)
            efficiency_score = f"{efficiency_score:.4f}"
        except:
            efficiency_score = "N/A"
        
        latex_code += f"{model} & {mode} & {duration} & {throughput} & {r20} & {mr20} & {efficiency_score} \\\\\n"
    
    # Close the table
    latex_code += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Efficiency Score = mR@20 / Duration (min). Higher values indicate better performance per unit time. Only successful runs (non-zero mR@20) are included.
\\end{tablenotes}
\\end{table}
"""
    
    return latex_code

def main():
    # Create all LaTeX tables
    full_latex = create_efficiency_latex_table()
    summary_latex = create_efficiency_summary_latex_table()
    performance_latex = create_performance_efficiency_latex_table()
    
    # Save all tables
    with open('data/benchmark/ag200/computational_efficiency_full.tex', 'w') as f:
        f.write(full_latex)
    
    with open('data/benchmark/ag200/computational_efficiency_summary.tex', 'w') as f:
        f.write(summary_latex)
    
    with open('data/benchmark/ag200/performance_efficiency_tradeoff.tex', 'w') as f:
        f.write(performance_latex)
    
    print("LaTeX efficiency tables generated:")
    print("1. computational_efficiency_full.tex - Complete efficiency table")
    print("2. computational_efficiency_summary.tex - Summary table with averages")
    print("3. performance_efficiency_tradeoff.tex - Performance vs efficiency trade-offs")
    
    print("\n" + "="*100)
    print("FULL EFFICIENCY LATEX TABLE:")
    print("="*100)
    print(full_latex)
    
    print("\n" + "="*100)
    print("SUMMARY EFFICIENCY LATEX TABLE:")
    print("="*100)
    print(summary_latex)
    
    print("\n" + "="*100)
    print("PERFORMANCE-EFFICIENCY TRADEOFF LATEX TABLE:")
    print("="*100)
    print(performance_latex)

if __name__ == "__main__":
    main()
