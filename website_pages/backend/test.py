import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pydicom
import io
from PIL import Image   # ✅ ADDED

# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(
    page_title="Intracranial Hemorrhage Detection",
    layout="wide"
)

st.title("🧠 Intracranial Hemorrhage Detection")

# ===============================
# Load Models (CACHED)
# ===============================
@st.cache_resource
def load_models():
    clf = tf.keras.models.load_model(
        "best_cls_model.keras",
        compile=False
        
    )
    seg = tf.keras.models.load_model(
        "ich_segmentation_unet.keras",
        compile=False
    )
    return clf, seg

clf_model, seg_model = load_models()

# ===============================
# Labels
# ===============================
SUBTYPES = [
    "Epidural",
    "Intraparenchymal",
    "Intraventricular",
    "Subarachnoid",
    "Subdural"
]

# ===============================
# Image Loader (DICOM / JPG / PNG) ✅ ADDED
# ===============================
def load_image(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".dcm"):
        ds = pydicom.dcmread(io.BytesIO(uploaded_file.read()), force=True)
        img = ds.pixel_array.astype(np.float32)
        return img, "dcm"

    else:  # jpg / png
        img = Image.open(uploaded_file).convert("L")
        img = np.array(img).astype(np.float32)
        return img, "img"

# ===============================
# Preprocessing (UNCHANGED LOGIC)
# ===============================
def preprocess_for_classification(img, size=224):
    img = np.clip(img, 0, 80)

    img -= img.min()
    img /= (img.max() + 1e-8)

    img = cv2.resize(img, (size, size))
    img = np.stack([img] * 3, axis=-1)

    return np.expand_dims(img, axis=0)

def preprocess_for_segmentation(img, size=256):
    img -= img.min()
    img /= (img.max() + 1e-8)

    img = cv2.resize(img, (size, size))
    img = np.stack([img] * 3, axis=-1)

    return img

# ===============================
# Upload (DICOM + JPG + PNG) ✅ MODIFIED
# ===============================
uploaded = st.file_uploader(
    "Upload Brain CT (.dcm / .jpg / .png)",
    type=["dcm", "jpg", "png"]
)

if uploaded:
    img, file_type = load_image(uploaded)

    # ===============================
    # Classification
    # ===============================
    cls_input = preprocess_for_classification(img)
    det_pred, cls_pred = clf_model.predict(cls_input, verbose=0)

    det_score = float(det_pred[0][0])
    predicted_subtype = SUBTYPES[np.argmax(cls_pred[0])]

    if det_score >= 0.5:
        st.error(f"🔴 Hemorrhage Detected (Probability: {det_score:.3f})")
    else:
        st.success(f"🟢 No Hemorrhage Detected (Probability: {det_score:.3f})")

    st.markdown(f"### Predicted Subtype: **{predicted_subtype}**")

    st.subheader("📊 Subtype Probabilities")
    for name, prob in zip(SUBTYPES, cls_pred[0]):
        st.write(f"{name}: {prob:.3f}")

    # ===============================
    # Localization (UNCHANGED)
    # ===============================
    if det_score >= 0.5:
        st.subheader("📍 Hemorrhage Localization (Confidence Heatmap)")

        seg_img = preprocess_for_segmentation(img)
        seg_input = np.expand_dims(seg_img, axis=0)

        prob_map = seg_model.predict(seg_input, verbose=0)[0, :, :, 0]

        high_conf = np.percentile(prob_map, 95)
        norm_map = np.clip(prob_map / (high_conf + 1e-6), 0, 1)

        heatmap = (norm_map * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        ct_gray = (seg_img[:, :, 0] * 255).astype(np.uint8)
        ct_gray = cv2.cvtColor(ct_gray, cv2.COLOR_GRAY2BGR)

        overlay = cv2.addWeighted(
            ct_gray, 0.6,
            heatmap_color, 0.6,
            0
        )

        binary_mask = (prob_map >= high_conf).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            overlay,
            contours,
            -1,
            (255, 255, 0),
            2
        )

        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                img,
                caption="Input CT Image",
                clamp=True
            )

        with col2:
            st.image(
                overlay,
                caption="Red = High Confidence | Yellow = Boundary",
                clamp=True
            )

        st.info(
            "Heatmap is confidence-scaled. "
            "Red regions indicate highest likelihood of hemorrhage."
        )

    else:
        st.info("Segmentation skipped (no hemorrhage detected).")
