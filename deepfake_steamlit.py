import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
import os
import warnings

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Deepfake Detection Dashboard",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Deepfake vs. Real Image Classifier")
st.write("Upload an image and provide its diagnostic metadata to evaluate if it is authentic or synthetic.")

# ═══════════════════════════════════════════════════════════════════════════
# 2. LOAD MODEL & PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_assets():
    model_path = "deepfake_nn_model.keras"
    pipeline_path = "preprocessing_pipeline.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file '{pipeline_path}' not found")

    model = tf.keras.models.load_model(model_path)
    pipeline = joblib.load(pipeline_path)
    print(f"✅ Model loaded. Input shape: {model.input_shape}")
    print(f"✅ Pipeline loaded")
    return model, pipeline

try:
    model, pipeline = load_assets()
except FileNotFoundError as e:
    st.error(f"❌ File Error: {str(e)}")
    st.info("📁 Make sure these files exist in the same directory:\n- deepfake_nn_model.keras\n- preprocessing_pipeline.pkl")
    st.stop()
except Exception as e:
    st.error(f"❌ Could not load model or pipeline: {str(e)}")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# 3. ⚠️  IMPORTANT: THESE MUST MATCH YOUR TRAINING DATA EXACTLY
# ═══════════════════════════════════════════════════════════════════════════
# From FINAL_DATASET.csv, after dropping metadata columns, the remaining
# feature columns (X) are:
#
#     gender            -> categorical  (Female, Male, Unknown)
#     age_group         -> categorical  (18-25, 26-35, 36-50, 50+)
#     image_quality     -> categorical  (High, Medium)   <-- NOT numeric!
#     confidence_score  -> numeric      (0.0 - 1.0)
#
# In training:
#     categorical_features = X.select_dtypes(include=['object']).columns
#     numerical_features   = X.select_dtypes(include=['int64','float64']).columns
#
# Because `image_quality` is a STRING column ("High"/"Medium"), it was
# routed into the OneHotEncoder branch, NOT the StandardScaler branch.
# The previous app version treated it as a 0.0-1.0 slider and fed it to
# StandardScaler (which was only fit on `confidence_score`) -> shape error.
#
# ⚠️  LABEL DIRECTION (also verified from the dataset):
#     label == 'REAL' -> label_numeric == 1
#     label == 'FAKE' -> label_numeric == 0
# So the model's sigmoid output is P(REAL), NOT P(FAKE).
# A HIGH score means REAL. A LOW score means FAKE.
# ═══════════════════════════════════════════════════════════════════════════

NUMERIC_COLS = ['confidence_score']
CATEGORICAL_COLS = ['gender', 'age_group', 'image_quality']

GENDER_OPTIONS = ['Female', 'Male', 'Unknown']
AGE_GROUP_OPTIONS = ['18-25', '26-35', '36-50', '50+']
IMAGE_QUALITY_OPTIONS = ['High', 'Medium']  # matches dataset exactly


def process_input_data(gender, age_group, image_quality, confidence_score):
    """
    Build a single-row DataFrame with the EXACT column names/dtypes used
    during training, then run it through the fitted pipeline as-is.
    No manual scaler/encoder extraction needed - just call .transform()
    on the full pipeline, the same way it was used during training.
    """
    try:
        input_df = pd.DataFrame({
            'gender': [gender],
            'age_group': [age_group],
            'image_quality': [image_quality],          # ✅ categorical string, not float
            'confidence_score': [float(confidence_score)]
        })

        print(f"Input DataFrame:\n{input_df}")
        print(f"dtypes:\n{input_df.dtypes}")

        # Use the pipeline exactly as fit during training (pipeline.fit_transform(X))
        processed_features = pipeline.transform(input_df)

        if hasattr(processed_features, "toarray"):
            processed_features = processed_features.toarray()

        if processed_features.ndim == 1:
            processed_features = processed_features.reshape(1, -1)

        print(f"✅ Transform successful. Shape: {processed_features.shape}")

        expected_features = model.input_shape[1]
        actual_features = processed_features.shape[1]

        if actual_features != expected_features:
            st.error(
                f"❌ Feature Mismatch! Model expects {expected_features} features, "
                f"got {actual_features}. Check that the pipeline used to save "
                f"`preprocessing_pipeline.pkl` matches the training script."
            )
            return None

        return processed_features

    except Exception as e:
        st.error(f"❌ Data processing error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 4. UI - IMAGE UPLOAD & METADATA
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("### Step 1: Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "gif"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image Preview", use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Width", f"{image.width}px")
        with col2:
            st.metric("Height", f"{image.height}px")
        with col3:
            st.metric("Format", image.format or "Unknown")
    except Exception as e:
        st.error(f"❌ Could not open image: {str(e)}")

    st.markdown("### Step 2: Enter Extraction Metadata")
    st.info("🔍 These must match the categories used during training.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Classification Factors")
        gender = st.selectbox("Gender Group", GENDER_OPTIONS)
        age_group = st.selectbox("Age Group", AGE_GROUP_OPTIONS)
        # ✅ FIXED: categorical dropdown, not a 0.0-1.0 slider
        image_quality = st.selectbox(
            "Image Quality",
            IMAGE_QUALITY_OPTIONS,
            help="Matches the 'image_quality' category from the dataset (High / Medium)"
        )

    with col2:
        st.subheader("🎯 Quality Metrics")
        confidence_score = st.slider(
            "Detection Confidence Score",
            min_value=0.0,
            max_value=1.0,
            value=0.90,
            step=0.01,
            help="This is the only true numeric feature in the dataset."
        )

    st.markdown("---")

    if st.button("🔮 Analyze Image Authenticity", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing image..."):
            processed_input = process_input_data(
                gender=gender,
                age_group=age_group,
                image_quality=image_quality,
                confidence_score=confidence_score
            )

            if processed_input is not None:
                try:
                    raw_output = model.predict(processed_input, verbose=0)[0]

                    if isinstance(raw_output, (list, np.ndarray)):
                        prediction_score = float(raw_output[0])
                    else:
                        prediction_score = float(raw_output)

                    prediction_score = float(np.clip(prediction_score, 0.0, 1.0))
                    print(f"Raw model output (P(REAL)): {prediction_score}")

                    # ✅ FIXED: label_numeric=1 means REAL, so HIGH score = REAL
                    is_real = prediction_score >= 0.5

                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")

                    if is_real:
                        st.success("### 🎉 Result: **REAL / AUTHENTIC**")
                        st.metric(
                            label="Authenticity Confidence",
                            value=f"{prediction_score * 100:.2f}%"
                        )
                        st.info("✅ This image appears to be authentic based on the analyzed features.")
                    else:
                        st.error("### 🚨 Result: **DEEPFAKE / SYNTHETIC**")
                        st.metric(
                            label="Deepfake Certainty",
                            value=f"{(1 - prediction_score) * 100:.2f}%"
                        )
                        st.warning("⚠️  This image appears to be a deepfake. Caution advised.")

                    st.markdown("### 📈 Detailed Metrics")
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Raw Model Score (P = REAL)", f"{prediction_score:.4f}")
                    with m2:
                        st.metric("Confidence Score Input", f"{confidence_score:.2f}")

                    st.markdown("---")
                    st.warning(
                        "⚠️  **Disclaimer**: Results are from a machine learning model trained on "
                        "metadata only (not raw pixel analysis) and may not be fully accurate."
                    )

                except Exception as err:
                    st.error(f"❌ Prediction Error: {str(err)}")
                    import traceback
                    print(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════
# 5. SIDEBAR INFO
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ℹ️ About This Tool")
    st.markdown("Deepfake detector using neural network classification on image metadata.")

    st.markdown("---")
    st.markdown("### 🔧 Technical Info")
    st.code(f"Input Shape: {model.input_shape}\nOutput Shape: {model.output_shape}")
    st.caption(
        "Feature layout: 1 numeric (confidence_score) + "
        "9 one-hot (gender[3] + age_group[4] + image_quality[2]) = 10 total"
    )

    st.markdown("---")
    st.markdown("### 📚 Label Convention")
    st.markdown(
        """
        - `label_numeric = 1` → **REAL**
        - `label_numeric = 0` → **FAKE**
        - Model output = P(REAL) — high score means real.
        """
    )