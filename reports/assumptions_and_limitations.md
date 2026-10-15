# Assumptions and Limitations

- Dataset 1 screening labels are an exact deterministic threshold on the supplied score/items; high accuracy is label reconstruction, not independent autism detection.
- Dataset 1 unique Case_No values do not prove distinct participants. There are no supplied linkage fields; records are provisionally treated as independent for splitting.
- Dataset 2 is described by its provider as synthetic/augmented. Class definitions and generation procedures are not clinically validated in supplied materials.
- Dataset 2 exact paths link WAV, motion JSON and physiology CSV to index sample IDs; they do not authenticate shared participant identity or timing.
- Dataset 2 image mapping is unresolved. No images are assigned to samples or used for modeling.
- Dataset licensing/reuse terms are unclear in both metadata files. Input archives remain outside this repository.
- Numeric modality checks are structural, not medical validation. No undocumented units or clinical meanings are inferred.
