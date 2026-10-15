# When Feature Engineering Outperforms Deep Learning: Sparsity-Aware Direct Multi-Step Forecasting for Small-Scale Retail Demand

**Manuscript submitted to Expert Systems with Applications (ESWA)**

This repository contains the LaTeX source, figures, submission materials, and the **complete reproducible forecasting code** for the paper.

## Abstract

Accurate multi-step demand forecasting for small-scale retail environments poses unique challenges: hundreds of SKUs with highly intermittent demand, short historical records (fewer than 200 days), and strong calendar effects create a regime where conventional deep learning models struggle to converge. We present a sparsity-aware direct multi-step forecasting pipeline evaluated on a real-world NYC coffee shop dataset (3 stores, 80 SKUs, 181 days). Our approach combines three key technical choices:

1. **Direct multi-step modeling** with independent LightGBM models for each forecast horizon (T+1 through T+7), eliminating recursive error propagation
2. **Tweedie loss** (variance power = 1.3) for robust learning under zero-inflated demand distributions
3. **Sparsity-aware feature engineering** that injects per-SKU characteristics (zero-rate, ADI, historical mean) into a unified global model

In a benchmark against DeepAR, PatchTST, and Temporal Fusion Transformer (TFT), our LightGBM-based approach with 53 engineered features achieves 23.0% lower MAE than DeepAR, 15.2% lower than PatchTST, and 22.9% lower than TFT. Ablation analysis reveals that direct multi-step modeling contributes the largest gain, while sparsity features improve both sparse (−4.8%) and dense (−1.1%) SKU segments simultaneously.

## Repository Structure

```
.
├── eswa_manuscript.tex              # Identified LaTeX manuscript (author record)
├── eswa_manuscript_anonymous.tex    # Anonymized version for peer review
├── eswa_title_page.tex / .txt       # Author info, affiliation, ORCID
├── eswa_cover_letter.txt            # Cover letter
├── eswa_highlights.txt              # 5 highlights (≤85 chars each)
├── eswa_credit_statement.txt        # CRediT author statement
├── eswa_declaration_of_interest.txt # Competing interests declaration
├── eswa_declaration_of_interest.docx
├── eswa_data_availability.txt       # Data availability statement
├── classical_baselines.py           # Croston / SBA / TSB baseline script
├── generate_fig1.py                 # Application context figure generator
├── generate_figures.py              # Experimental result figure generators
├── package_eswa_submission.py       # Submission packaging & validation script
├── convert_to_eswa.py               # Draft → ESWA format converter
├── SUBMISSION_CHECKLIST.md          # Editorial Manager upload checklist
├── figures/                         # 6 manuscript figures
│   ├── fig1_application_context.png
│   ├── fig1_overall_comparison.png
│   ├── fig2_ablation.png
│   ├── fig3_horizon_improvement.png
│   ├── fig4_segment_performance.png
│   ├── fig5_per_window_stability.png
│   └── fig6_direction_and_r2.png
├── data/
│   ├── daily_sku_sales.csv          # 240 SKU × 181 days panel (43,440 rows)
│   └── rolling_backtest_predictions.csv  # LightGBM predictions + SHAP
├── submission/                      # Final packaged submission (PDF + figures)
└── forecast/                        # Full forecasting project source code
    ├── predict_pipeline.py
    ├── backtest_pipeline.py
    ├── production_sparse.py
    ├── first_round_model_experiments.py … fourth_round_sparse_features.py
    ├── gluonts_bench.py             # DeepAR / PatchTST / TFT benchmarks
    ├── requirements.txt
    ├── Coffee Shop Sales.xlsx       # Raw dataset
    ├── nyc_weather_2023_h1.csv      # Weather + holiday features
    ├── dashboard/                   # Interactive SHAP dashboard
    ├── reports/model_quality/       # All experimental results
    └── test_*.py                    # Regression tests
```

## Compiling the Manuscript

The manuscript uses the Elsevier `elsarticle` class with Harvard (author-year) referencing. Compile with `pdfLaTeX` (BibTeX not needed — references use `\bibitem`):

```bash
pdflatex eswa_manuscript.tex
pdflatex eswa_manuscript.tex   # run twice for cross-references
```

For the anonymous review version:

```bash
pdflatex eswa_manuscript_anonymous.tex
pdflatex eswa_manuscript_anonymous.tex
```

Alternatively, upload the `.tex` files to [Overleaf](https://www.overleaf.com/) and select the **pdfLaTeX** compiler.

## Reproducing Experiments

### 1. Install dependencies

```bash
cd forecast
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Train production model + generate predictions
python3 predict_pipeline.py

# 5-window rolling backtest with baselines
python3 backtest_pipeline.py

# Winning experiment (Round 4: sparsity labels)
python3 fourth_round_sparse_features.py
```

### 3. Run deep learning baselines

```bash
python3 gluonts_bench.py   # DeepAR, PatchTST, TFT on CPU
```

### 4. Run classical baselines (Croston / SBA / TSB)

```bash
python3 classical_baselines.py --series data/daily_sku_sales.csv
```

### 5. Regenerate figures

```bash
python3 generate_fig1.py
python3 generate_figures.py
```

### 6. Run tests

```bash
cd forecast
python3 -m pytest test_*.py -v
```

## Key Results

| Method | MAE | WAPE | R² | Direction Hit |
|--------|:---:|:---:|:---:|:---:|
| **LightGBM + Tweedie + sparsity features** | **2.411** | **0.347** | **0.661** | **61.4%** |
| Recursive L1 baseline | 2.633 | — | — | 53.6% |
| Croston | 2.726 | 0.418 | 0.538 | 60.3% |
| SBA | 2.756 | 0.422 | 0.523 | 59.5% |
| TSB | 2.735 | 0.419 | 0.537 | 60.3% |
| DeepAR | 3.13 | — | — | — |
| PatchTST | 2.84 | — | — | — |
| TFT | 3.14 | — | — | — |

## Submission Package

Run the packaging script to validate and assemble the submission:

```bash
python3 package_eswa_submission.py
```

This checks:
- Abstract ≤ 250 words
- Each highlight ≤ 85 characters
- No author details in the anonymous manuscript
- All citations and figure references resolved

The packaged output goes to `submission/`.

## Author

**Zhanpeng Qin** — ByteDance, Beijing, China
- Email: richardandelu50@gmail.com
- ORCID: 0009-0008-9819-8831

## License

The code is provided for research reproducibility. The Coffee Shop Sales dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/ahmedabbas757/coffee-sales).
