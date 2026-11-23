# MECO clip-level clustering: results summary

**Notebook:** `MECO_clip_clustering.ipynb` (run on Colab, sklearn 1.4.2, numpy 1.26.4, Python 3.12.10)
**Input:** 42 subjects, 30,592 four-second windows of 35 OpenFace AUs
**Unit of analysis:** clip (one stimulus block), 690 clips after quality control
**Felt label:** self-reported emotion (6 categories)
**Clustering objective:** k = 6, fixed in advance, k-means on PCA of 175 window statistics averaged per clip

## Bottom line

The clusters do not recover the emotions people reported. Agreement with the self-reports is at chance
level. Two other findings matter more than the agreement number:

1. Part of the cluster structure follows *who* the person is, not what they felt.
2. The data has quality problems (missing clips, long clips, unbalanced felt labels) that need checking
   before any further modelling.

## 1. Data quality

| Item | Result |
|---|---|
| Raw clips | 733 (protocol: 756 = 42 subjects × 3 sessions × 6 clips) |
| Sessions failing the protocol | 8 of 126: S13 s3 (4 clips), S14 s3 (3), S17 s3 (3), S20 s3 (3), S35 s3 (2), S36 s2 (5 clips), S37 s1 (3), S38 s1 (2) |
| Pattern across failing sessions | 5 of 8 are session 3 |
| Clips excluded for session failures | 25 (733 → 708) |
| Clips dropped because most windows failed face detection | 18 (708 → 690) |
| Windows kept | 26,923 of 30,592 (88%) |
| Clip length (windows) | min 20, median 39 (≈156 s), max 88 (≈352 s) |
| Clips longer than 60 windows (protocol max, 4 min) | to be checked: possible merged segments |

**Felt labels are unbalanced.** The stimulus is balanced at about 115 clips per emotion. The self-reports are not:

| Category | Stimulus clips | Self-reported clips |
|---|---|---|
| angry | 114 | 96 |
| boredom | 114 | 50 |
| happiness | 118 | 147 |
| neutral | 115 | 161 |
| sadness | 114 | 113 |
| tension | 115 | 123 |

Boredom was rarely self-reported, which matches the paper's decision to merge it into neutral. Participants
often reported neutral or happy when a sad or bored clip was shown, so the felt label is noisier than the
stimulus label.

## 2. Clustering results (clip level, n = 690)

| Analysis | Result | Interpretation |
|---|---|---|
| k = 6 vs felt label, ARI | 0.004 | Essentially zero |
| k = 6, NMI | 0.016 | Essentially zero |
| k = 6, purity | 0.267 | Majority-class baseline is 0.233 (neutral) |
| k = 6, permutation null p-value | 0.061 | Not significant at 0.05 |
| Chi-square, clusters × felt label | χ² = 39.1, df = 25, p = 0.036 | Weak association |
| Data-driven k | 2 | Silhouette 0.155 at k = 2, about 0.06–0.08 for k ≥ 3 |
| Stability (mean ARI between restarts) | k = 2: 0.98; k = 3: 0.94; k = 4: 0.83; k = 6: 0.79; k = 10: 0.58 | Only the two-way split is clearly reproducible |
| HDBSCAN | 100% noise | No dense groups |
| ARI(clusters, subject) | 0.045 | Small overall, but see Section 3 |

**Cluster sizes (k = 6):** 111, 125, 118, 35, 191, 110.

**Cluster contents (self-report shares):**

| Cluster | angry | boredom | happiness | neutral | sadness | tension |
|---|---|---|---|---|---|---|
| 0 | 0.16 | 0.05 | 0.20 | 0.31 | 0.16 | 0.13 |
| 1 | 0.11 | 0.06 | 0.34 | 0.18 | 0.14 | 0.17 |
| 2 | 0.13 | 0.10 | 0.19 | 0.23 | 0.17 | 0.18 |
| 3 | 0.14 | 0.03 | 0.31 | 0.26 | 0.06 | 0.20 |
| 4 | 0.17 | 0.06 | 0.14 | 0.25 | 0.18 | 0.20 |
| 5 | 0.11 | 0.13 | 0.20 | 0.18 | 0.18 | 0.20 |

**Only notable enrichment:** cluster 1 is 34% happiness against a baseline of 21% (binomial p = 0.001,
uncorrected for 36 cells). It could be real, but it needs replication before anyone relies on it.

## 3. Identity leaks into the clusters

The clip-level t-SNE map shows tight islands of three to six clips from the same person (for example
video_S2, video_S39, video_S28, video_S30). A cluster that follows identity cannot separate emotions.

**Likely cause:** the per-subject z-scoring is applied to the AU intensities, but the 18 presence columns
are binary and were left unnormalised. Each person's presence rates are stable traits, so they survive into
the features.

**Test:** centre the presence columns per subject as well, rerun, and check whether the islands shrink.

## 4. Window-level check

ARI = 0.010, p = 0.000 (2,000 permutations). Statistically detectable, practically negligible. The small
p-value is partly because windows within a clip are correlated and long clips contribute many rows. Report
the clip-level result.

## 5. Not yet analysed

- **The two-way split is unexplained.** Check whether it tracks mean AU activation, the share of failed
  frames, or clip length. A split driven by face visibility would be a quality issue, not an emotion signal.
- **Stimulus agreement is not computed.** Earlier, window-level agreement between stimulus and self-report
  was about 0.74. The clip-level ARI between stimulus and felt labels is the realistic ceiling for any
  face-based method and should be reported next to the clustering ARI.
- **Valence and arousal are not tested.** A Kruskal-Wallis test of valence across clusters is the next step.

## 6. Recommended next steps, in order

1. Centre the presence columns per subject and rerun. Check whether the subject islands shrink.
2. Explain the k = 2 split using mean AU activation, failed-frame share and clip length.
3. Compute the stimulus-vs-felt ARI at clip level as the reference.
4. Run the valence and arousal tests across clusters.
5. Check the eight sessions with missing clips against the raw recordings (five are session 3).
6. Check clips longer than 60 windows.

If step 1 removes the islands and the k = 6 agreement stays at chance, the honest conclusion is that AU
statistics over 4-second windows do not separate the felt categories in this dataset.
