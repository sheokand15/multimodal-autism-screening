# Machine Learning Models

## Summary

The project contains two separate experimental tracks. Track A reconstructs a questionnaire dataset's screening-trait label. Track B demonstrates multimodal classification and fusion on a small, provider-described synthetic dataset. Neither track validates a clinical autism screening system.

| Track | Inputs | Models | Research question |
|---|---|---|---|
| A — questionnaire | Ten binary questionnaire item codes | Transparent score threshold, multinomial logistic regression, CART decision tree | Can the supplied target label be reconstructed from its item codes? |
| B — multimodal | Audio, motion, and physiology summary features | Modality-specific multinomial logistic regression; early- and late-fusion logistic regression | Can simple feature-level and decision-level fusion be demonstrated? |

## Track A

### Score-rule reference

The positive label is generated when `sum(A1..A10) > 3`. This transparent rule is a reference baseline, not a learned AI model.

### Logistic regression

The ten binary item codes form the feature matrix. A NumPy batch-gradient-descent multinomial logistic regression model is trained with regularization. Scaling is fit on training rows only; regularization is selected by validation macro-F1. The model's class probabilities are uncalibrated.

### CART decision tree

A small Gini-impurity classification tree provides a nonlinear comparison. Maximum depth is selected on the validation partition, with a minimum leaf size constraint.

### Interpretation

The dataset target exactly matches the score threshold computed from A1-A10. The 100% held-out accuracy reported for the rule and logistic model therefore measures reconstruction of a deterministic target. It is not 100% autism detection, diagnosis, or clinical screening accuracy.

## Track B

### Modality branches

Each supported modality is represented by engineered summary features:
- Audio: 10 features.
- Motion: 63 features.
- Physiology: 21 features.

A regularized multinomial logistic-regression classifier is trained for each branch. Image data are omitted because the index-to-image mapping is unresolved.

### Early fusion

Early fusion concatenates the audio, motion, and physiology vectors into a joint feature representation, then trains one multinomial logistic regression classifier. This is feature-level fusion: modalities are combined before classification.

### Late fusion

Late fusion trains one classifier per modality and takes the equal-weight mean of their class-probability vectors, followed by normalization. This is decision-level fusion. If the demo accepts multiple uploaded files, they are assumed to represent the same sample; identity and synchronization are not verified.

## Evaluation protocol

- Random seed: 42.
- Stratified train/validation/test partitions: nominally 70%/15%/15%.
- Observed Track A counts: 738/158/158.
- Observed Track B counts: 70/15/15.
- Hyperparameter selection: validation macro-F1.
- Final evaluation: reserved test partition.
- Stability diagnostic: two repeats of five-fold stratified cross-validation on development rows (train plus validation), not on the held-out test data.
- Metrics include accuracy, balanced accuracy, and macro-F1. Interpret them alongside test size and target definition.

## What is implemented

The current learners are classical machine-learning models over engineered tabular features. Fusion consists of feature concatenation and equal-weight probability averaging. The implementation does **not** include a Transformer encoder, self-attention, a trainable fusion gate, a pretrained foundation model, or an autonomous agent. Describing the current system with those labels would be inaccurate.

## Performance and limitations

Track A's perfect label reconstruction is expected from the deterministic target construction. Track B has 100 indexed examples, 70 training examples, and 15 test examples; provider metadata describes its data as synthetic/augmented, and current test performance is low and highly uncertain. Neither track supports generalization to real participants or clinical use.

Adding a higher-capacity architecture alone does not establish better performance. Future Transformer or attention-based fusion experiments would require substantially more well-documented, participant-independent data and a pre-specified evaluation protocol. Compare any future model with these baselines using identical leakage controls, ablations, and uncertainty reporting.
