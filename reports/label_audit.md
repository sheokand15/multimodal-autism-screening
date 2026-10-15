# Label Audit

## Toddler questionnaire dataset

The actual target header is `Class/ASD Traits ` (trailing space), with `Yes` (728) and `No` (326). Supplied metadata documents a positive label for Q-CHAT-10 score greater than 3 and says the class is assigned automatically by the screening application. In the archive, `Qchat-10-Score` equals the sum of A1–A10 for all 1,054 records, and target equals `score > 3` for all 1,054. Score and target are mechanically linked.

The target represents the dataset’s screening-trait label, not a confirmed clinical diagnosis. The score is excluded as a predictor; A1–A10 may assess reconstruction of that same rule, with the rule itself reported as the transparent baseline.

## Multimodal dataset

The index contains 100 labels: typical 25, mild_asd 23, moderate_asd 19, severe_asd 33. Supplied metadata describes a partially synthetic and augmented dataset and explicitly disclaims real individuals and clinical use. Treat these as simulated class labels. File-level correlations may reflect the generation recipe, duplicates, templates, or class-specific artifacts; they do not establish behavioral or physiological biomarkers.
