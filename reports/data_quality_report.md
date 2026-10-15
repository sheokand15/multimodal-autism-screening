# Data Quality Report

Automated read-only audit. Input archives were not modified.

## Dataset 1 — toddler questionnaire
- File: `Toddler Autism dataset July 2018.csv`; rows: 1054; columns: 19.
- Actual target header: `Class/ASD Traits ` (trailing whitespace preserved in source).
- Missing cells: 0; duplicate rows: 0; unique Case_No values: 1054.
- Labels: {'No': 326, 'Yes': 728}.
- Age_Mons range: 12–36.
- Qchat-10-Score distribution: {3: 96, 4: 110, 10: 75, 9: 95, 8: 97, 5: 120, 6: 96, 2: 88, 0: 54, 7: 135, 1: 88}.
- Item-positive counts: {'A1': 594, 'A2': 473, 'A3': 423, 'A4': 540, 'A5': 553, 'A6': 608, 'A7': 685, 'A8': 484, 'A9': 516, 'A10': 618}.
- A1–A10 response values outside 0/1: 0.
- `Qchat-10-Score == sum(A1..A10)`: 1054/1054.
- Target equals (`score > 3`): 1054/1054.

### Dataset 1 split feasibility
- train: 738 records; class counts {'No': 228, 'Yes': 510}.
- validation: 158 records; class counts {'Yes': 109, 'No': 49}.
- test: 158 records; class counts {'Yes': 109, 'No': 49}.

Categorical distributions:
- `Sex`: {'f': 319, 'm': 735}
- `Ethnicity`: {'middle eastern': 188, 'White European': 334, 'Hispanic': 40, 'black': 53, 'asian': 299, 'south asian': 60, 'Native Indian': 3, 'Others': 35, 'Latino': 26, 'mixed': 8, 'Pacifica': 8}
- `Jaundice`: {'yes': 288, 'no': 766}
- `Family_mem_with_ASD`: {'no': 884, 'yes': 170}
- `Who completed the test`: {'family member': 1018, 'Health Care Professional': 24, 'Health care professional': 5, 'Self': 4, 'Others': 3}

The score and target are excluded from ML predictors. Case_No is an identifier and excluded. No participant linkage is supplied.

## Dataset 2 — multimodal synthetic dataset
- Index rows: 100; labels: {'mild_asd': 23, 'moderate_asd': 19, 'severe_asd': 33, 'typical': 25}; unique sample IDs: 100.
- Exact indexed paths resolving: image 0/100, audio 100/100, motion 100/100, physiology 100/100.
- Images: decoded 100/100; width range 120–799, height range 148–947; RGB 100/100; exact duplicate groups 0; dHash near-duplicate candidate pairs (Hamming distance ≤4): 0; decode errors 0.
- WAV: decoded 100/100; rates {16000: 100}; channels {1: 100}; durations min/median/max 2.000/2.000/2.000s; silence fraction median 0.00127; clipped fraction max 0.00000; exact duplicate groups 0; max absolute downsampled waveform correlation 0.935; pairs at |r|≥0.95: 0; errors 0.
- Motion JSON: decoded 100/100; schemas {('duration_sec', 'frame_rate', 'head', 'left_hand', 'right_hand', 'stimming_detected', 'torso'): 100}; sequence lengths {(300, 300, 300, 300): 100}; numeric range -0.4562 to 0.4337; nonfinite values 0; duplicate groups 0; errors 0.
- Physiology CSV: decoded 100/100; row counts {100: 100}; column schemas {('Time', 'HR', 'GSR', 'TEMP'): 100}; constant column counts {}; duplicate groups 0; errors 0.
- Physiology time ordering: 100/100 files strictly increase; interval min/median/max 0.101010/0.101010/0.101010 seconds.
- Raw value ranges as stored (units undocumented): {'HR': (78.9125390099638, 100.935813096624), 'GSR': (1.64200685667231, 2.64197197879683), 'TEMP': (37.0, 37.0587785252293)}.

### Interpretation and mapping
- Both archives passed `ZipFile.testzip()` (no CRC errors).
- Exact sample-ID references resolve audio, motion and physiology paths; they do not independently verify participant provenance or synchronization.
- All image paths from the index are unresolved: index references `images/child_NNN.png`, while actual files are `images/Child (n).jpg`. No filename-position matching was performed. Image paths remain blank in the manifest.
- Dataset metadata explicitly describes generated/synthetic or augmented content. Labels are simulated dataset labels and cannot establish clinical associations.
- Each physiology CSV has `Time`, `HR`, `GSR`, `TEMP` and 100 records from 0 to 10 seconds. The entire TEMP series is identical across all samples (1 unique sequences). Units and medical meanings are not inferred.
- Split feasibility: one indexed record per sample ID, with 19–33 examples per class. A stratified file-level 70/15/15 partition is feasible, but cannot be described as participant-level clinical generalization.
- Dataset 2 manifest uses a fixed seed-derived stratified 70/15/15 split by sample_id; exact assignments are saved.
- train: 70 rows; class counts {'mild_asd': 16, 'moderate_asd': 13, 'severe_asd': 23, 'typical': 18}.
- validation: 15 rows; class counts {'severe_asd': 5, 'mild_asd': 3, 'typical': 4, 'moderate_asd': 3}.
- test: 15 rows; class counts {'typical': 3, 'mild_asd': 4, 'severe_asd': 5, 'moderate_asd': 3}.
