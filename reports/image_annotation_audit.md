# Image and Annotation Audit

## Files and decoding

The archive has 100 decodable RGB JPEG images (`images/Child (n).jpg`), with dimensions varying from 120×148 to 799×947 pixels. No exact duplicates, dHash near-duplicate candidates (Hamming distance ≤4), or decode errors were found. A visual review of a contact sheet showed close-up portrait photographs of children with varied backgrounds, poses, and facial expressions. This visual appearance cannot establish whether images depict real people or how the images were sourced; provider metadata describes the collection as simulated/augmented.

## Mapping

The official index and embedded metadata refer to `images/child_NNN.png`. No such files exist in the archive. The actual JPEG names do not contain the indexed IDs. The 100 images cannot be joined to participants using supplied documentation, hashes, embedded metadata, or other authoritative keys. No positional or alphabetical association was created. Image paths are blank in the track B manifest, and image-based training and image-inclusive fusion are excluded.

## Annotation schema and use

No image-specific bounding boxes, landmarks, segmentation masks, action units, gaze labels, or expert-reviewed behavioral annotations were provided. The index provides only sample-level simulated class labels. Those labels cannot be treated as image annotations even if an image join is later established. A future image study would require verified mapping, a stated image-level labeling protocol, and human review. Facial appearance, emotion, eye contact, and gaze are not individually diagnostic of autism.

## Decision

The image modality is not used for modeling or demo inference. The present dataset supports no image-specific claim.
