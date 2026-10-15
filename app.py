"""Optional Streamlit research demo; inference remains non-clinical."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path

import streamlit as st

from src.config import PROJECT_ROOT
from src.inference.predict import predict_multimodal_files, predict_questionnaire

st.set_page_config(page_title="Multimodal Screening Research Demo", layout="wide")
st.title("Early Autism Screening — Research Prototype")
st.warning("Research and educational demonstration only. Outputs reproduce dataset labels and are not clinical diagnoses or medical advice.")
page = st.sidebar.radio("Page", ["Questionnaire", "Multimodal research demo", "Model evaluation", "Dataset quality"])

if page == "Questionnaire":
    st.header("Questionnaire coding demonstration")
    st.write("Enter the binary item codes used in the supplied dataset. A1–A9 use 1 for Sometimes/Rarely/Never; A10 uses 1 for Always/Usually/Sometimes. This screen does not administer the clinical instrument.")
    item_labels = [
        "Looks at people when spoken to",
        "Points to indicate wants",
        "Brings objects to show you",
        "Responds to your smile or facial expression",
        "Imitates your actions or gestures",
        "Responds when you call his/her name",
        "Engages in pretend play",
        "Follows your gaze",
        "Shows objects to share interest",
        "Uses simple gestures",
    ]
    responses = [int(st.checkbox(f"A{i}: {item_labels[i - 1]}", key=f"a{i}")) for i in range(1, 11)]
    if st.button("Calculate research output"):
        result = predict_questionnaire(responses)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Transparent scoring rule")
            st.metric("Q-CHAT-coded score", result["screening_rule"]["score"])
            st.write(f"Dataset screening-trait output: **{result['screening_rule']['screening_trait_output']}** (positive when score > 3).")
        with col2:
            st.subheader("Learned model (separate output)")
            st.write(f"Dataset screening-trait prediction: **{result['learned_model']['screening_trait_prediction']}**")
            st.json(result["learned_model"]["uncalibrated_class_probabilities"])
            st.caption("Probabilities are uncalibrated. The model reproduces a score-derived dataset label; it does not independently detect autism.")

elif page == "Multimodal research demo":
    st.header("Synthetic multimodal file prototype")
    st.write("Upload any supported subset of files. If combining files, ensure they belong to the same sample; the prototype cannot verify identity or synchronization. Images are not accepted because the supplied image-to-sample mapping is unresolved.")
    audio = st.file_uploader("Audio WAV (mono 16-bit PCM)", type=["wav"])
    motion = st.file_uploader("Motion sequence JSON", type=["json"])
    physiology = st.file_uploader("Physiology time-series CSV", type=["csv"])
    if st.button("Run research prototype"):
        if not any((audio, motion, physiology)):
            st.error("Upload at least one supported modality.")
        else:
            paths = []
            try:
                kwargs = {}
                for key, upload in (("audio_path", audio), ("motion_path", motion), ("physiology_path", physiology)):
                    if upload is not None:
                        suffix = {"audio_path": ".wav", "motion_path": ".json", "physiology_path": ".csv"}[key]
                        temp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                        temp.write(upload.getvalue()); temp.close(); paths.append(Path(temp.name)); kwargs[key] = temp.name
                result = predict_multimodal_files(**kwargs)
                st.write("Available modalities:", ", ".join(result["available_modalities"]))
                st.subheader("Branch predictions")
                st.json(result["branch_predictions"])
                st.subheader("Late fusion over available branches")
                st.json(result["selected_fusion_prediction"])
                if "early_fusion_prediction" in result:
                    st.subheader("Early fusion (requires all three supported modalities)")
                    st.json(result["early_fusion_prediction"])
                st.caption("Probabilities are uncalibrated. This model was trained on the provider-described synthetic dataset; class labels and modalities are not clinically validated.")
            except Exception as error:
                st.error(f"Could not process these files: {error}")
            finally:
                for path in paths:
                    path.unlink(missing_ok=True)

elif page == "Model evaluation":
    st.header("Held-out research results")
    a = json.loads((PROJECT_ROOT / "artifacts/dataset1_results.json").read_text(encoding="utf-8"))
    b = json.loads((PROJECT_ROOT / "artifacts/dataset2_results.json").read_text(encoding="utf-8"))
    st.subheader("Dataset 1 — questionnaire screening-trait reconstruction")
    st.dataframe([{"model": name, "accuracy": values["accuracy"], "balanced accuracy": values["balanced_accuracy"], "macro F1": values["macro_f1"], "test n": values["n"]} for name, values in a.items()], hide_index=True)
    st.caption("A1–A10 exactly determine the dataset target through score > 3. These results are not independent autism detection.")
    st.subheader("Dataset 2 — synthetic file-level benchmark")
    result_rows = [{"model": name, "accuracy": values["accuracy"], "balanced accuracy": values["balanced_accuracy"], "macro F1": values["macro_f1"], "test n": values["n"]} for name, values in b["unimodal"].items()]
    result_rows += [{"model": name, "accuracy": b[name]["accuracy"], "balanced accuracy": b[name]["balanced_accuracy"], "macro F1": b[name]["macro_f1"], "test n": b[name]["n"]} for name in ("early_fusion", "late_fusion")]
    st.dataframe(result_rows, hide_index=True)
    st.caption("The test set has 15 samples. Performance is highly uncertain and does not support clinical or real-participant generalization.")

else:
    st.header("Dataset quality and missing modalities")
    for file in ("data_quality_report.md", "label_audit.md", "cross_modal_consistency_report.md", "image_annotation_audit.md", "assumptions_and_limitations.md"):
        path = PROJECT_ROOT / "reports" / file
        if path.exists():
            with st.expander(file, expanded=file in ("data_quality_report.md", "label_audit.md")):
                st.markdown(path.read_text(encoding="utf-8"))
    st.write("Dataset 2: 100/100 index rows have audio, motion, and physiology files resolvable by exact paths. Image index paths resolve for 0/100 rows, so images are omitted from modeling.")
