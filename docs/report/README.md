# Technical report

IEEE conference-format paper covering the methodological findings of this
project.

## Build

Requires a LaTeX distribution with `IEEEtran` (MiKTeX or TeX Live).

```
pdflatex f1_tyre_degradation.tex
pdflatex f1_tyre_degradation.tex     # second pass resolves cross-references
```

Produces a five-page PDF. The second pass is required; without it,
`\ref` targets render as `??`.

## Contents

The paper reports four findings, in decreasing order of transferability
beyond motorsport:

1. **Correlation-based filtering of regression slopes selects on magnitude,
   not precision.** A `|r|` threshold applied to a fitted slope cannot retain
   a genuinely flat relationship, because `r → 0` as the slope does,
   independently of how small the residual scatter is. This is a general
   property of the statistic, not a quirk of this dataset.
2. **A precision-based replacement**, bounding the standard error of the
   slope, and the measured effect of the substitution.
3. **The standard `mean(v²)` energy proxy correlates negatively** with
   degradation and is better described as a track geometry descriptor.
4. **Improvement thresholds are uninterpretable without a measured ceiling.**

Every number in the paper is reproducible from the scripts listed in the
Reproducibility section, and is recorded with more context in
`../FINDINGS.md`.

## Note on scope

The paper reports a model that **fails** its pre-registered acceptance
criterion, and says so. The intellectual content is in the diagnosis of why,
not in the model's performance.
