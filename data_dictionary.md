# Data Dictionary

## Track A — toddler questionnaire CSV

| Field | Meaning / handling |
|---|---|
| `Case_No` | Row identifier; excluded from predictors. |
| `A1`–`A10` | Dataset’s binary Q-CHAT-10 item codes; model predictors. Item values are not raw response text. |
| `Age_Mons` | Age in months; observed range 12–36; not used in the final model. |
| `Qchat-10-Score` | Sum of A1–A10; excluded as a target-derived feature. |
| `Sex`, `Ethnicity` | Supplied demographics; not used as predictors. |
| `Jaundice` | Supplied yes/no field; exact clinical measurement details unavailable. |
| `Family_mem_with_ASD` | Supplied family-history field; questionnaire wording/source may be imperfect. |
| `Who completed the test` | Screening-app form field. |
| `Class/ASD Traits ` | Target with trailing space in actual CSV; values `Yes`/`No`; exactly `score > 3`. It is a screening-trait label, not a clinical diagnosis. |

The source DOCX says A1–A9 receive code 1 for Sometimes/Rarely/Never and A10 receives code 1 for Always/Usually/Sometimes, with a positive screening-trait output when the total exceeds 3.

## Track B — multimodal index

| Field | Meaning / handling |
|---|---|
| `sample_id` | Index sample identifier; used as split group and file stem for audio, motion, and physiology. |
| `label` | Provider class among `typical`, `mild_asd`, `moderate_asd`, `severe_asd`; simulated labels according to supplied metadata. |
| `image` | Indexed PNG path; does not resolve to actual JPG archive names; excluded from modeling. |
| `voice` | Mono 16-bit PCM WAV at 16 kHz, about 2 seconds; synthetic per metadata. |
| `motion` | JSON with four 300×3 position sequences and metadata fields. Coordinate units/semantics undocumented. |
| `physio` | 100-row CSV with `Time`, `HR`, `GSR`, `TEMP`; units and device provenance undocumented. Temperature sequence repeats exactly across all 100 samples. |

The index links files by package-level names but does not independently establish real participant identity or synchronized collection.
