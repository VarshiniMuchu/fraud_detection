from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
EXP_CSV = OUT / 'experiment_results.csv'
SYN_CSV = OUT / 'synthetic_quality_report.csv'
PDF_PATH = OUT / 'model_output_report.pdf'


def clean_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def add_title_page(pdf, title, subtitle):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    ax.text(0.5, 0.82, title, ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(0.5, 0.74, subtitle, ha='center', va='center', fontsize=12, color='dimgray')

    box = dict(boxstyle='round,pad=0.5', facecolor='#f2f2f2', edgecolor='#bdbdbd')
    ax.text(0.5, 0.58, 'Summary based on project output files', ha='center', va='center', fontsize=11, bbox=box)
    ax.text(0.5, 0.48, 'Includes experiment metrics, ensemble performance, autoencoder-aware results, and CTGAN quality.', ha='center', va='center', fontsize=10)
    ax.text(0.5, 0.32, 'Project: Fraud Detection Research', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.24, 'Source files: output/experiment_results.csv and output/synthetic_quality_report.csv', ha='center', va='center', fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def add_experiment_summary_page(pdf, exp_df):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')

    ax.text(0.5, 0.96, 'Experiment Results Summary', ha='center', va='top', fontsize=16, fontweight='bold')

    display_df = exp_df[['Experiment', 'Model', 'PR-AUC', 'F1-Score', 'Recall', 'ROC-AUC', 'FPR (%)', 'FNR (%)', 'Simulated Cost ($)']].copy()
    display_df['PR-AUC'] = display_df['PR-AUC'].map(lambda x: f'{float(x):.4f}')
    display_df['F1-Score'] = display_df['F1-Score'].map(lambda x: f'{float(x):.4f}')
    display_df['Recall'] = display_df['Recall'].map(lambda x: f'{float(x):.4f}')
    display_df['ROC-AUC'] = display_df['ROC-AUC'].map(lambda x: f'{float(x):.4f}')
    display_df['FPR (%)'] = display_df['FPR (%)'].map(lambda x: f'{float(x):.4f}')
    display_df['FNR (%)'] = display_df['FNR (%)'].map(lambda x: f'{float(x):.4f}')
    display_df['Simulated Cost ($)'] = display_df['Simulated Cost ($)'].map(lambda x: f'{float(x):.0f}')

    table = Table(ax, bbox=[0.02, 0.14, 0.96, 0.76])
    ncols = len(display_df.columns)
    nrows = len(display_df.index) + 1
    width = 0.96 / ncols

    for (i, col) in enumerate(display_df.columns):
        cell = table.add_cell(0, i, width, 0.05, text=col, loc='center', facecolor='#d9e8ff')
        cell.set_fontsize(8)
        cell.set_text_props(weight='bold')

    for r in range(len(display_df.index)):
        for c in range(ncols):
            value = str(display_df.iloc[r, c])
            cell = table.add_cell(r + 1, c, width, 0.05, text=value, loc='center')
            cell.set_fontsize(7)

    ax.add_table(table)

    # Small bar chart for PR-AUC
    ax2 = fig.add_axes([0.12, 0.02, 0.8, 0.12])
    ax2.bar([i for i in range(len(exp_df))], exp_df['PR-AUC'].astype(float), color='#4c78a8')
    ax2.set_xticks(range(len(exp_df)))
    ax2.set_xticklabels(exp_df['Experiment'].str.replace('Exp ', 'E').tolist(), rotation=45, ha='right', fontsize=6)
    ax2.set_ylabel('PR-AUC', fontsize=8)
    ax2.set_title('PR-AUC by Experiment', fontsize=9)

    pdf.savefig(fig)
    plt.close(fig)


def add_ctgan_quality_page(pdf, syn_df):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.5, 0.96, 'CTGAN Synthetic Data Quality', ha='center', va='top', fontsize=16, fontweight='bold')

    selected = syn_df[syn_df['category'].isin(['statistical_similarity', 'correlation_preservation', 'privacy'])].copy()
    selected = selected[['category', 'metric', 'value']].reset_index(drop=True)
    selected['value'] = selected['value'].map(lambda x: f'{float(x):.4f}' if str(x).replace('.', '', 1).replace('-', '', 1).replace('e', '', 1).replace('+', '', 1).isdigit() else str(x))

    table = Table(ax, bbox=[0.12, 0.28, 0.76, 0.62])
    columns = ['Category', 'Metric', 'Value']
    rows = selected.values.tolist()
    ncols = 3
    nrows = len(rows) + 1
    width = 0.76 / ncols

    for i, col in enumerate(columns):
        cell = table.add_cell(0, i, width, 0.05, text=col, loc='center', facecolor='#d9e8ff')
        cell.set_fontsize(9)
        cell.set_text_props(weight='bold')

    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = table.add_cell(r, c, width, 0.05, text=str(val), loc='center')
            cell.set_fontsize(8)

    ax.add_table(table)

    # Utility metrics section
    utility = syn_df[syn_df['category'] == 'tstr_utility'].copy()
    utility = utility[['metric', 'value']].reset_index(drop=True)
    utility['value'] = utility['value'].map(lambda x: f'{float(x):.4f}')

    ax.text(0.12, 0.18, 'TSTR / TRTR utility highlights:', fontsize=10, fontweight='bold')
    text = ''
    for _, row in utility.iterrows():
        text += f"{row['metric']}: {row['value']}\n"
    ax.text(0.12, 0.12, text, fontsize=8, va='top', ha='left')

    pdf.savefig(fig)
    plt.close(fig)


def add_summary_page(pdf, exp_df):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')

    best = exp_df.sort_values('PR-AUC', ascending=False).iloc[0]
    ax.text(0.5, 0.9, 'Simple Interpretation', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(0.08, 0.76, 'Best overall result:', fontsize=11, fontweight='bold')
    ax.text(0.08, 0.70, f"- {best['Experiment']} | {best['Model']}", fontsize=10)
    ax.text(0.08, 0.64, f"- PR-AUC: {float(best['PR-AUC']):.4f}", fontsize=10)
    ax.text(0.08, 0.58, f"- F1-Score: {float(best['F1-Score']):.4f}", fontsize=10)
    ax.text(0.08, 0.52, f"- Recall: {float(best['Recall']):.4f}", fontsize=10)
    ax.text(0.08, 0.46, f"- Simulated cost: ${float(best['Simulated Cost ($)']):.0f}", fontsize=10)

    ax.text(0.08, 0.36, 'What this means:', fontsize=11, fontweight='bold')
    ax.text(0.08, 0.30, 'The full hybrid framework did best overall. It improved recall and reduced false negatives.', fontsize=10)
    ax.text(0.08, 0.24, 'The autoencoder-aware model also improved the fraud-detection balance versus the baseline.', fontsize=10)
    ax.text(0.08, 0.18, 'CTGAN helped create more balanced training data, but synthetic quality still shows some differences from real fraud data.', fontsize=10)

    pdf.savefig(fig)
    plt.close(fig)


def main():
    if not EXP_CSV.exists():
        raise FileNotFoundError(f'Missing experiment results file: {EXP_CSV}')
    if not SYN_CSV.exists():
        raise FileNotFoundError(f'Missing synthetic quality file: {SYN_CSV}')

    exp_df = pd.read_csv(EXP_CSV)
    syn_df = pd.read_csv(SYN_CSV)

    with PdfPages(str(PDF_PATH)) as pdf:
        add_title_page(pdf, 'Fraud Detection Model Output Summary', 'Model metrics, ensemble results, CTGAN quality, and interpretation')
        add_experiment_summary_page(pdf, exp_df)
        add_ctgan_quality_page(pdf, syn_df)
        add_summary_page(pdf, exp_df)

    print(f'[+] PDF report saved to: {PDF_PATH}')


if __name__ == '__main__':
    main()
