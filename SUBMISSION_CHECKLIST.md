# ESWA Editorial Manager checklist

Journal: Expert Systems with Applications (Elsevier)
Title: When Feature Engineering Outperforms Deep Learning: Sparsity-Aware Direct Multi-Step Forecasting for Small-Scale Retail Demand
Sole and corresponding author: Zhanpeng Qin
Affiliation: ByteDance, Beijing, China
Email: richardandelu50@gmail.com
ORCID: 0009-0008-9819-8831
Date: 14 October 2026
Review type: double anonymized

This file is a guide for the person uploading. Do not upload the checklist itself.

## Limits already checked

- Abstract: **225 words** (limit 250). Count is whitespace-delimited words after LaTeX commands are removed. Hyphenated compounds such as "multi-step" count as one word. Splitting those hyphens gives 242 words, still within the limit.
- Highlights (`eswa_highlights.txt`), each line is one highlight and includes spaces:

  | Characters | Highlight |
  | ---: | --- |
  | 71 | Direct multi-step Tweedie LightGBM cuts MAE 8.5% vs. recursive baseline |
  | 66 | Sparsity features improve sparse (-4.8%) and dense (-1.1%) SKU MAE |
  | 64 | 23.0%, 15.2%, and 22.9% lower MAE than DeepAR, PatchTST, and TFT |
  | 74 | Conformal P10-P90 intervals cover 80.6% of actuals; TreeSHAP explains them |

- Keywords (6): retail demand forecasting; direct multi-step prediction; Tweedie loss; sparsity-aware features; LightGBM; TreeSHAP explainability.
- Re-run `python3 package_eswa_submission.py` before upload. It rejects an abstract over 250 words, a highlight over 85 characters, author details inside the anonymized manuscript or its PDF, and unresolved citation or figure references.

## What each file is

| File | Role |
| --- | --- |
| `eswa_manuscript.tex` | Identified author record. Keep this off the review upload. |
| `eswa_manuscript_anonymous.tex` / `.pdf` | Manuscript for reviewers. |
| `eswa_title_page.tex` / `.txt` / `.pdf` | Author, affiliation, email, ORCID, acknowledgments, funding, competing interests, CRediT, and the code URL. |
| `eswa_highlights.txt` | Four highlights, ready to paste. |
| `eswa_cover_letter.txt` | Sole-author letter (I / my). |
| `eswa_credit_statement.txt` | CRediT for Zhanpeng Qin. |
| `eswa_declaration_of_interest.txt` and `.docx` | Singular "The author declares..." with the nothing-to-declare box marked. |
| `eswa_data_availability.txt` | Commercial confidentiality, aggregated statistics on request, public code URL. |
| `submission/` | Flat copies of the upload files and the six figures used in the paper. |

## Upload order

1. **Cover letter.** Paste or upload `eswa_cover_letter.txt`.
2. **Title page.** Upload `submission/eswa_title_page.pdf` (source: `eswa_title_page.tex`). This page carries the author name and is not part of the review file.
3. **Anonymized manuscript.** Upload `submission/eswa_manuscript_anonymous.pdf` as the manuscript PDF, and the LaTeX source below with it. Do not upload `eswa_manuscript.tex`.
4. **Highlights.** Paste the four lines from `eswa_highlights.txt` into the highlights box.
5. **CRediT statement.** Paste `eswa_credit_statement.txt`.
6. **Declaration of competing interests.** Upload `eswa_declaration_of_interest.docx`. Editorial Manager prefers the Word file from the Elsevier declarations tool (https://declarations.elsevier.com/). The docx here is the standard nothing-to-declare statement in the singular. If EM rejects it, regenerate that form in the tool and select no competing interests. `eswa_declaration_of_interest.txt` is the same text.
7. **Data availability statement.** Paste `eswa_data_availability.txt`. It includes `https://github.com/qzpthuhhu/eswa-retail-forecasting`. The anonymized manuscript does not include that URL, because the account name can identify the author. In the review file the statement is: raw data cannot be shared due to commercial confidentiality; aggregated statistics are available upon reasonable request; code will be made available upon publication.
8. **Figures.** If EM asks for figures as separate items, upload the six PNGs in `submission/` (flat names, no folder):
   - `fig1_application_context.png`
   - `fig1_overall_comparison.png`
   - `fig2_ablation.png`
   - `fig3_horizon_improvement.png`
   - `fig4_segment_performance.png`
   - `fig6_direction_and_r2.png`
9. **Double-anonymized review.** Reviewers should receive only the anonymized manuscript. The title page, cover letter, CRediT statement, declaration, and the data statement that contains the GitHub URL stay with the editorial office.
10. **Remaining manual steps.**
    - Suggested reviewers are optional.
    - SSRN / preprint opt-in is optional. An identified public preprint during review can defeat double-anonymized review, so leave that opt-in off unless a preprint is intended.
    - Link ORCID `0009-0008-9819-8831` on the EM author profile.
    - If EM has a separate funding box, enter: "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors."
    - Confirm one author, marked as corresponding author.

## Flat LaTeX package

`submission/` has no subfolders. Figure paths inside the anonymized `.tex` are bare filenames (`fig1_application_context.png` and so on). `\graphicspath{{}{figures/}}` also finds the copies under `figures/` when the same `.tex` is compiled from the repository root.

To make one archive with files at the zip root:

```bash
cd submission && zip -r ../eswa_em_submission.zip .
```

Upload that zip as the LaTeX source if EM accepts a single archive. The archive must not contain `eswa_manuscript.tex`.

PNG figures are embedded in the compiled manuscript PDF. If EM refuses PNG uploads, export the same images to TIFF or EPS without changing the plots.

## P0 revisions (14 October 2026)

The scientific text is the same in `eswa_manuscript.tex` and `eswa_manuscript_anonymous.tex`. The review file still has no author name, affiliation, email, ORCID, funding sentence, or code URL. Shengchao Qin is a coauthor of Riachy et al. (2025), so the review bibliography contains that surname. `package_eswa_submission.py` checks Zhanpeng, ByteDance, the email, the ORCID, and the GitHub account. It does not treat the substring "qin" as an identity leak.

- **Bibliography.** 41 entries. Journal DOIs added in this revision were read from Crossref on 14 October 2026. Three conference items already in the manuscript (LightGBM, PatchTST, N-BEATS) still have no DOI. Syntetos, Babai, and Gardner (2015) keeps the previous volume and pages; a guessed DOI was a different paper, so no DOI was added. Related Work and Table 1 (positioning) discuss Riachy et al. (2025), Vallés-Pérez et al. (2022), Long et al. (2025), Nathan et al. (2026), and Yılmaz Sucuoğlu et al. (2026), and separate this paper from Nathan and from Long: small daily coffee panel, direct horizons 1--7, sparsity covariates versus a split, and an explicit DeepAR/PatchTST/TFT comparison.
- **Metrics.** The proposed model is best on MAE (2.411), R² (0.661), and direction hit (61.4%). Split models are best on WAPE (0.432 versus 0.452) and on sparse MAE (0.432 versus 0.452). MAE is the primary metric. The text states that WAPE is proportional to MAE on a shared sample, and that the four gradient-boosted WAPE entries match the sparse-MAE column. Those cells were not recomputed, and no error was replaced.
- **Deployment.** The conclusion, the discussion, and the cover letter describe the point forecast, the conformal P10--P90 interval (80.6% coverage), and TreeSHAP. They do not say the system is in production and they do not claim a what-if study.
- **Croston, SBA, TSB.** The protocol, the smoothing constants (α = 0.1, and β = 0.1 for TSB), and the recursions are in the manuscript. Every cell in that table is TODO. `python3 classical_baselines.py --self-check` checks the algebra. `python3 classical_baselines.py` exits 2 and writes no metrics, because `data/daily_sku_sales.csv` is not in the repository. No MAE was invented. The comparison figures were not redrawn.
- **Limits.** Abstract still **225 words**. The four highlights are unchanged and still within 85 characters. Re-run `python3 package_eswa_submission.py` after any further edit; it recompiled the review PDF and the title page for this revision.

## Left out on purpose

- `figures/fig5_per_window_stability.png` is not cited and is not in `submission/`. In `generate_figures.py` the series is marked as simulated placeholder data ("would come from actual backtest"), so the figure is not part of the reported results.
- The manuscript previously cited `fig:arch`, but no architecture figure exists. That cross-reference was removed so the PDF does not show "Figure ??". The five pipeline stages in that sentence are unchanged.
- Bibliography correction: Ben Taieb and Atiya, "A bias and variance analysis for multistep-ahead time series forecasting," was entered as 1999 in *IEEE Transactions on Neural Networks*. The article is Ben Taieb and Atiya (2016), *IEEE Transactions on Neural Networks and Learning Systems*, 27, 62–76. The citation key and Harvard label match that year. No DOI was added.
