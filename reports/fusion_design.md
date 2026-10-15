# Fusion Design

## Available inputs

The image mapping is unresolved, so only audio, motion, and physiology features are used. Each branch uses compact, deterministic summary features: signal-level acoustic statistics, coordinate/derivative statistics over the motion sequences, and value/time-series statistics for the three named physiology columns. The feature definitions and fitted normalization are saved with model artifacts. Training-only normalization is applied to validation and test data.

## Implemented prototypes

- **Unimodal branches:** regularized multinomial logistic regression for each of audio, motion, and physiology. Regularization is selected on validation macro-F1.
- **Early fusion:** concatenate the three feature vectors, standardize using training samples, and fit a regularized multinomial logistic model. This is a small feature-level fusion baseline, not a learned neural encoder.
- **Late fusion:** average the three branch probability vectors with equal weights. This fixed, transparent rule is assessed on validation and test data.

The dataset has only 70 training samples and four classes. It is partially synthetic, pairing provenance is not independently verified, and images cannot be linked. Attention-based intermediate fusion, cross-modal transformers, pretrained vision/audio encoders, and trainable gating would add parameters without a defensible sample size or verified image input. They are not implemented. The simple prototypes are intended to demonstrate interfaces and experimental bookkeeping only.

## Limitations

The output class names are provider-supplied simulation labels, not clinical severity ratings. Model performance must not be interpreted as clinical utility or as evidence that physiological, voice, or movement properties diagnose autism. The held-out set has 15 samples, so the observed scores are highly uncertain. Late fusion may dilute a useful branch; early fusion may overfit. No calibration or uncertainty method is reported as clinically meaningful.
