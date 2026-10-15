# Dataset Inventory — Milestone 1

Inspection date: 2026-10-13 (Asia/Calcutta)

## Workspace

The supplied workspace directory contained no project files and is not initialized as a Git repository. No existing source code, environment configuration, or repository guidance was found. This milestone adds documentation only. The source archives and external metadata were read in place and were not modified or extracted into the workspace.

## Archive integrity and contents

Both ZIP archives were opened and every member was read through Python `zipfile.testzip()`; each returned `None` (no CRC failure detected).

| Input archive | Size | Files | Contents observed |
|---|---:|---:|---|
| `autism-screening-for-toddlers-data.zip` | 83,665 bytes | 3 | Two CSV files and `Toddler data description.docx` |
| `multimodal-autism-dataset-zip.zip` | 13,606,289 bytes | 402 | Index CSV, embedded metadata JSON, 100 JPG images, 100 WAV audio files, 100 motion JSON files, and 100 physiology CSV files |

### Dataset 1: Toddler questionnaire

The archive contains `Toddler Autism dataset July 2018.csv` (1,054 records, 19 columns) and `Autism_Screening_Data_Combined.csv` (6,075 records, 15 columns). The requested toddler file has `Case_No`, A1–A10, `Age_Mons`, `Qchat-10-Score`, `Sex`, `Ethnicity`, `Jaundice`, `Family_mem_with_ASD`, `Who completed the test`, and a target column whose actual header is `Class/ASD Traits ` (trailing space). Its observed labels are `Yes` (728) and `No` (326); all cells are populated, and all 1,054 rows are unique. `Case_No` is unique for all rows.

Observed `Age_Mons` spans 12–36. A1–A10 sum to `Qchat-10-Score` in all 1,054 rows. The target equals `Yes` exactly when the score is greater than 3 in all 1,054 rows. This is direct evidence that the supplied label reconstructs the questionnaire screening rule; it is not evidence of clinical diagnosis. The dataset metadata says the class was automatically assigned by the ASDTests screening app based on the score.

The metadata describes Q-CHAT-10 item coding and a `>3` positive-trait threshold. It lists the license as `Unknown`. The accompanying DOCX data dictionary was read from the archive and corroborates that score threshold and automatic class assignment. No provenance beyond the supplied Kaggle metadata and its cited sources is independently verified here.

### Dataset 2: Multimodal behavioral dataset

The archive contains `autism_dataset_index.csv` with 100 rows and columns `sample_id`, `image`, `voice`, `motion`, `physio`, `label`. Labels observed in that index: `typical` 25, `mild_asd` 23, `moderate_asd` 19, `severe_asd` 33. The same 100 sample descriptions are in the embedded `autism_dataset_metadata.json` (a JSON list).

For each indexed sample, the index names one WAV (`voice/child_NNN.wav`), one JSON (`motion/child_NNN.json`), one physiology CSV (`physio/child_NNN.csv`), and one PNG (`images/child_NNN.png`). The archive actually contains the expected 100 WAVs, JSONs, and physiology CSVs under those ID-based paths. It contains 100 JPG images named `images/Child (n).jpg`; it does not contain any indexed PNG paths. The image-to-sample mapping is unresolved; no image was paired by sort order or presumed numeric correspondence. Dataset 2 has 100 rows with all four paths populated in the index, but only the three non-image modality paths resolve by exact name. 100/100 samples have those three available modalities and 0/100 have a verified four-modality mapping. Participant identity and recording synchronization remain unverified.

The external metadata describes the dataset as partially synthetic and augmented, including simulated images, synthetic speech, simulated motion and physiological series. It states that the dataset does not represent real individuals and is not for clinical use. Thus the classes are dataset-provided simulation labels, not verified clinical categories. The external Croissant metadata gives a license description of `Other (specified in description)`; the exact reuse terms need clarification from the dataset source before redistribution. Metadata names Kaggle as the catalog and `Kadhiravan UCET` as creator.

One sample physiology CSV has columns `Time`, `HR`, `GSR`, `TEMP` with 100 rows over nominally 0–10 seconds. One sample motion JSON has `head` coordinate arrays; a full schema/quality audit remains for the next milestone. Image decoding/dimensions, WAV technical properties/content, duplicate hashes, signal quality, complete cross-modal validation, and exact modality coverage remain unverified.

## Input metadata and limitations

The two user-supplied JSON metadata files are Croissant-style records. Dataset 1 metadata says 1,054 toddler records and 18 attributes including the class, but the actual target CSV has 19 columns including `Case_No`; use the archive as the source of actual file schema. Dataset 2 metadata describes 100 samples and four modalities, but its described PNG filenames disagree with actual JPG archive filenames. The metadata cannot by itself establish the correctness of image assignments or clinical validity.

## Reproducibility note

Commands were run with the bundled Python runtime located via Codex workspace dependencies. No model training was run. No raw data or generated models were copied into the repository.
