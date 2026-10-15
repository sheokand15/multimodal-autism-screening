# Milestone 1 — Feasibility Assessment

## Findings

1. **Questionnaire experiment is feasible as a screening-rule reconstruction study.** The expected 1,054 toddler rows and ten binary items are present. The target is exactly determined by `sum(A1..A10) > 3`; the score field is exactly that sum. Consequently, the score and target must not be predictors in a learned classifier. A transparent score-rule baseline is appropriate. Classifiers may be compared only as methods for reproducing the screening-trait label, and high scores must not be framed as independent autism detection. The unique `Case_No` can support deterministic record-level splitting but should be excluded as a predictor.
2. **Dataset 2 is suitable for a cautious synthetic-data pipeline prototype.** The index defines four classes and ID-based links for WAV, JSON and physiology files. Those three path families exist. This dataset can support engineering demonstrations on simulated data if structural/data checks confirm the mappings and no class-generating artifacts make evaluation tautological. It cannot support claims about real children or clinical screening.
3. **Image-based work is blocked by an unresolved index/archive mismatch.** The official index points to `images/child_NNN.png`; the archive holds `images/Child (n).jpg`. No verified mapping is presently available. Image experiments and image-containing fusion should be excluded until an authoritative mapping or embedded metadata/hash establishes pairings. Alphabetic or ordinal alignment is not acceptable.
4. **Cross-modal pairing and independence are not yet established.** Matching IDs in the index for WAV/JSON/physiology provide a file naming link, not proof that files are independently collected participant records or synchronized modalities. Metadata calls the data partially synthetic/augmented. Group splitting by `sample_id` is the only defensible initial split unit if the dataset records are distinct; provenance and synthetic generation patterns need auditing before interpreting held-out metrics.
5. **Reuse terms are uncertain.** Dataset 1 metadata lists an unknown license; Dataset 2 indicates other terms specified in its description. Keep inputs local and do not redistribute them in project outputs until terms are resolved.

## Recommended order

- Next, extract data only into a clearly ignored/local raw-data location (or read from archives directly), inspect the DOCX data dictionary, run the full tabular and modality-level quality audits, and write machine-readable manifests with unresolved image paths explicitly flagged.
- Complete and document Dataset 1 score/target audit and stratified fixed-seed split design before fitting any estimator.
- For Dataset 2, audit all file decodes, audio headers/waveforms, JSON sequences, physiology series, duplicates, label/feature relationships and ID consistency. Build splits by sample ID only after establishing independent units. Consider non-image experiments only if the data pass audit; use language such as “synthetic benchmark” and clearly state the limits.
- Do not train models until both tracks' data audit and split strategy are documented, as requested in the project brief.

## Commands and execution evidence

- Read both external JSON metadata files.
- Opened both ZIPs and ran `zipfile.testzip()` across members; no CRC errors were reported.
- Counted archive files and modality extensions; read the Dataset 2 index and labels; inspected example motion JSON and physiology CSV; read Dataset 1 CSV schemas, record counts, missingness, uniqueness, score sums and target threshold relation.
- No training, test execution, extraction, or data mutation was performed.

## Unresolved

- Dataset 2 image mapping; image content and decode checks.
- Audio decode/content and data quality.
- Full motion/physiology schemas, signal checks and generated-label artifacts.
- Dataset 1 DOCX data dictionary review; detailed feature/category/age audit.
- Exact dataset reuse licenses and independent-participant provenance.
