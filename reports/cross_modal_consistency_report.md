# Cross-Modal Consistency Report

## Index and path checks

The dataset index contains 100 unique `sample_id` values. For every ID, the named WAV, motion JSON, and physiology CSV exists at the exact path specified by the index. All 100 audio files decode as mono 16-bit PCM, 16 kHz, and approximately 2 seconds. All 100 motion JSON files parse with the same four joint arrays (`head`, `left_hand`, `right_hand`, `torso`), each 300×3, plus `frame_rate`, `duration_sec`, and `stimming_detected`. All 100 physiology CSVs parse with `Time`, `HR`, `GSR`, and `TEMP`, 100 rows spanning 0–10 seconds. No exact duplicate hashes were found within each modality family.

The 100 JPGs decode as RGB, but none matches the index's PNG path. Thus 0/100 samples have all four modalities verified by exact index paths; 100/100 have all three non-image file paths. Every sample is incomplete only with respect to the unverified image modality. The manifest marks image paths unresolved and includes an exclusion reason. No cross-modal image association was inferred.

## Signal observations

- WAV headers are consistent (16 kHz, mono, 2 seconds). Median near-silence fraction is 0.00127 and maximum clipped-sample fraction is zero. Comparing downsampled, centered waveforms found no pair with absolute correlation ≥0.95 (maximum 0.935); exact file hashes also found no duplicates. The archive metadata characterizes them as synthetic child speech; speech content was not independently transcribed or validated.
- Motion sequence schemas and lengths are consistent. `frame_rate` is 30 and `duration_sec` is 10 in every file. Coordinates are simulated numeric sequences; no physical units or clinically meaningful kinematics are documented.
- Image duplicate checks found no exact hashes or dHash near-duplicate candidates at Hamming distance ≤4. No image class-wise or participant-wise distribution can be verified without the missing mapping.
- Physiology sequences share the same 100-row time grid and column names. `TEMP` has the same mean across samples at the observed precision and appears highly templated; units are undocumented. `HR` and `GSR` vary, but their variation is narrow across labels. No medical interpretation is assigned.

## Provenance and pairing

ID-consistent filenames show that the dataset publisher intended the audio, motion, and physiology files to correspond to each index sample. The supplied materials do not independently establish that these are synchronized measurements from the same real participant or session. In fact, the external metadata describes the dataset as partially synthetic and augmented and says it does not represent real individuals. Therefore, exact paths establish package-level linkage only. Any held-out evaluation is reported as a synthetic file-level benchmark, not participant-level or clinical validation.

## Split

The fixed seed-42 stratified 70/15/15 split is saved in `data/manifests/track_b_manifest.csv`, grouping each row by its unique sample ID. It yields 70 train, 15 validation, and 15 test samples. This is suitable for a small implementation demonstration, but the test set is too small to support stable per-class conclusions.
