#!/usr/bin/env python3
"""
Convert the mR@K table to LaTeX format.
"""

import pandas as pd
import os


def create_latex_table():
    """Create LaTeX table from the CSV data."""

    # Read the CSV file
    csv_file = "data/benchmark/ag200/mr_table.csv"
    df = pd.read_csv(csv_file)

    # Create LaTeX table
    latex_code = """
\\begin{table}[htbp]
\\centering
\\caption{Comprehensive mR@K Results for All Models and Modes on AG200 Dataset}
\\label{tab:mr_results}
\\begin{tabular}{lcccccccc}
\\toprule
\\textbf{Model} & \\textbf{Mode} & \\textbf{R@10} & \\textbf{R@20} & \\textbf{R@50} & \\textbf{R@100} & \\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50} & \\textbf{mR@100} \\\\
\\midrule
"""

    # Add data rows
    for _, row in df.iterrows():
        model = row["Model"]
        mode = row["Mode"]
        r10 = row["R@10"]
        r20 = row["R@20"]
        r50 = row["R@50"]
        r100 = row["R@100"]
        mr10 = row["MR@10"]
        mr20 = row["MR@20"]
        mr50 = row["MR@50"]
        mr100 = row["MR@100"]

        latex_code += f"{model} & {mode} & {r10} & {r20} & {r50} & {r100} & {mr10} & {mr20} & {mr50} & {mr100} \\\\\n"

    # Close the table
    latex_code += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Results are shown for all three models (STTRAN, DSG-DETR, TEMPURA) across three evaluation modes (predcls, sgcls, sgdet) on the AG200 dataset. Training was performed for 10 epochs with 20\% of the dataset (fraction=5). Note that STTRAN and DSG-DETR show zero performance on sgcls mode, indicating potential training issues.
\\end{tablenotes}
\\end{table}
"""

    return latex_code


def create_summary_latex_table():
    """Create a summary LaTeX table with key metrics."""

    # Read the CSV file
    csv_file = "data/benchmark/ag200/mr_table.csv"
    df = pd.read_csv(csv_file)

    # Create summary table focusing on mR@20 and mR@100
    latex_code = """
\\begin{table}[htbp]
\\centering
\\caption{Summary of mR@20 and mR@100 Results by Model and Mode}
\\label{tab:mr_summary}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Model} & \\textbf{Mode} & \\textbf{mR@20} & \\textbf{mR@100} \\\\
\\midrule
"""

    # Group by model for better readability
    models = ["STTRAN", "DSG-DETR", "TEMPURA"]
    modes = ["predcls", "sgcls", "sgdet"]

    for model in models:
        model_data = df[df["Model"] == model]
        first_row = True

        for mode in modes:
            mode_data = model_data[model_data["Mode"] == mode]
            if not mode_data.empty:
                mr20 = mode_data.iloc[0]["MR@20"]
                mr100 = mode_data.iloc[0]["MR@100"]

                if first_row:
                    latex_code += f"\\multirow{{3}}{{*}}{{{model}}} & {mode} & {mr20} & {mr100} \\\\\n"
                    first_row = False
                else:
                    latex_code += f" & {mode} & {mr20} & {mr100} \\\\\n"

    # Close the table
    latex_code += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Best performing model per mode: predcls (DSG-DETR: 0.1493), sgcls (TEMPURA: 0.1401), sgdet (TEMPURA: 0.0811).
\\end{tablenotes}
\\end{table}
"""

    return latex_code


def main():
    # Create full LaTeX table
    full_latex = create_latex_table()

    # Create summary LaTeX table
    summary_latex = create_summary_latex_table()

    # Save both tables
    with open("data/benchmark/ag200/mr_table_full.tex", "w") as f:
        f.write(full_latex)

    with open("data/benchmark/ag200/mr_table_summary.tex", "w") as f:
        f.write(summary_latex)

    print("LaTeX tables generated:")
    print("1. mr_table_full.tex - Complete table with all metrics")
    print("2. mr_table_summary.tex - Summary table with key metrics")

    print("\n" + "=" * 80)
    print("FULL LATEX TABLE:")
    print("=" * 80)
    print(full_latex)

    print("\n" + "=" * 80)
    print("SUMMARY LATEX TABLE:")
    print("=" * 80)
    print(summary_latex)


if __name__ == "__main__":
    main()
