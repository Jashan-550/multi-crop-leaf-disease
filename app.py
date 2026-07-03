"""
TomatoGuard-CNN — Streamlit web interface for tomato leaf disease classification.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from disease_info import DISEASE_INFO, DISCLAIMER

APP_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    APP_DIR / "best_model.keras",
    APP_DIR / "results" / "best_model.keras",
]
LABEL_MAP_CANDIDATES = [
    APP_DIR / "label_map.json",
    APP_DIR / "results" / "label_map.json",
]
REPORTED_TEST_ACCURACY = 0.938


def resolve_existing_path(candidates: list[Path]) -> Path | None:
    """Return the first existing path from a list of candidates."""
    for path in candidates:
        if path.is_file():
            return path
    return None


@st.cache_resource
def load_model() -> tf.keras.Model | None:
    """Load the trained Keras model (preprocessing is baked into the graph)."""
    model_path = resolve_existing_path(MODEL_CANDIDATES)
    if model_path is None:
        return None
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_label_map() -> dict | None:
    """Load class index mapping and model metadata."""
    for label_path in LABEL_MAP_CANDIDATES:
        if not label_path.is_file():
            continue
        with label_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("index_to_class"):
            return data
    return None


def build_index_to_class(label_map: dict) -> dict[int, str]:
    """Build a clean int-keyed class lookup from label_map JSON (keys are strings)."""
    return {int(k): v for k, v in label_map["index_to_class"].items()}


def preprocess_image(image: Image.Image, img_size: tuple[int, int]) -> np.ndarray:
    """
    Prepare a PIL image for model inference.

    IMPORTANT: The model has EfficientNet preprocessing baked in.
    Feed RAW [0, 255] float32 RGB values — do NOT normalize or rescale.
    """
    width, height = img_size
    rgb = image.convert("RGB")
    resized = rgb.resize((width, height))
    array = np.asarray(resized, dtype=np.float32)  # RAW [0, 255]

    expected_shape = (height, width, 3)
    if array.shape != expected_shape:
        raise ValueError(
            f"Image array shape {array.shape} does not match expected {expected_shape}. "
            "Ensure the upload is a valid RGB image."
        )

    return np.expand_dims(array, axis=0)


def lookup_class(index_to_class: dict[int, str], idx: int | np.integer) -> str:
    """Map a model output index to a class name using the int-keyed dict."""
    return index_to_class[int(idx)]


def display_name(class_key: str) -> str:
    """Convert raw class key to a human-readable display name."""
    if class_key == "Tomato___healthy":
        return "Healthy"

    name = class_key.replace("Tomato___", "")
    name = name.replace("_", " ")

    # Collapse duplicated leading crop word (e.g. "Tomato mosaic virus" -> "Mosaic Virus")
    if name.lower().startswith("tomato "):
        name = name[7:]

    if "spider mite" in name.lower():
        return "Spider Mites (Two-Spotted)"

    return name.title()


def render_html(markup: str) -> None:
    """
    Render raw HTML via st.markdown.

    Streamlit's markdown parser converts any line that follows a blank line and
    is indented 4+ spaces into a literal code block. Stripping per-line
    indentation and dropping blank lines keeps the whole fragment as one
    contiguous HTML block so it renders as markup, not code.
    """
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def inject_css() -> None:
    """Inject global custom CSS for a clean plant-health aesthetic."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        .stApp {
            background: #f7faf7;
        }

        .block-container {
            max-width: 720px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, h4 {
            letter-spacing: -0.02em;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e2ebe2;
        }

        section[data-testid="stSidebar"] .block-container {
            max-width: 100%;
            padding-top: 1.5rem;
        }

        .tg-sidebar-panel {
            background: #f7faf7;
            border: 1px solid #e2ebe2;
            border-radius: 14px;
            padding: 1.1rem 1.15rem;
            margin-bottom: 1rem;
        }

        .tg-sidebar-panel h3 {
            margin: 0 0 0.65rem 0;
            font-size: 0.95rem;
            font-weight: 700;
            color: #1b3a1f;
        }

        .tg-sidebar-panel p,
        .tg-sidebar-panel li {
            margin: 0.35rem 0;
            font-size: 0.88rem;
            line-height: 1.55;
            color: #5f6b62;
        }

        .tg-sidebar-meta {
            display: grid;
            gap: 0.45rem;
        }

        .tg-sidebar-meta span {
            display: block;
            font-size: 0.84rem;
            color: #5f6b62;
        }

        .tg-sidebar-meta strong {
            color: #2e7d32;
            font-weight: 600;
        }

        .tg-disclaimer {
            font-size: 0.78rem;
            line-height: 1.5;
            color: #7a8680;
            margin-top: 0.5rem;
        }

        .tg-header {
            text-align: center;
            margin: 0.5rem 0 1.75rem 0;
        }

        .tg-header-icon {
            font-size: 1.35rem;
            margin-bottom: 0.35rem;
        }

        .tg-header h1 {
            margin: 0;
            font-size: 1.85rem;
            font-weight: 700;
            color: #1b3a1f;
        }

        .tg-header p {
            margin: 0.45rem 0 0 0;
            font-size: 1rem;
            color: #5f6b62;
        }

        .tg-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, #c8dcc8, transparent);
            margin: 1.5rem 0;
            border: none;
        }

        .tg-empty-state {
            background: #ffffff;
            border: 1px dashed #b8d4ba;
            border-radius: 16px;
            padding: 2rem 1.5rem;
            text-align: center;
            color: #5f6b62;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
            margin: 0.5rem 0 1.25rem 0;
        }

        .tg-empty-state-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
            opacity: 0.7;
        }

        .tg-empty-state p {
            margin: 0;
            font-size: 0.95rem;
        }

        div[data-testid="stFileUploader"] {
            background: #ffffff;
            border: 2px dashed #66bb6a;
            border-radius: 16px;
            padding: 0.75rem 0.5rem 1rem 0.5rem;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
            transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
        }

        div[data-testid="stFileUploader"]:hover {
            border-color: #2e7d32;
            background: #f0f7f0;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        }

        div[data-testid="stFileUploader"] label p {
            font-size: 0.95rem !important;
            color: #1b3a1f !important;
            font-weight: 600 !important;
        }

        div[data-testid="stFileUploader"] small {
            color: #5f6b62 !important;
        }

        .tg-image-wrap {
            display: flex;
            justify-content: center;
            margin: 1.25rem 0 1.5rem 0;
        }

        .tg-image-wrap img {
            max-width: 300px;
            width: 100%;
            border-radius: 16px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
            border: 3px solid #ffffff;
        }

        div[data-testid="stImage"] {
            display: flex;
            justify-content: center;
        }

        div[data-testid="stImage"] img {
            max-width: 300px !important;
            border-radius: 16px !important;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06) !important;
            border: 3px solid #ffffff !important;
        }

        div[data-testid="stImage"] div[data-testid="caption"],
        div[data-testid="stImage"] figcaption {
            text-align: center;
            color: #7a8680;
            font-size: 0.82rem;
        }

        div[data-testid="stButton"] {
            display: flex;
            justify-content: center;
        }

        div[data-testid="stButton"] > button {
            width: 100%;
            max-width: 760px;
            background: linear-gradient(135deg, #2e7d32 0%, #388e3c 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 1.5rem !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.01em;
            box-shadow: 0 4px 14px rgba(46, 125, 50, 0.28);
            transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
        }

        div[data-testid="stButton"] > button:hover {
            background: linear-gradient(135deg, #256628 0%, #2e7d32 100%) !important;
            box-shadow: 0 6px 20px rgba(46, 125, 50, 0.35);
            transform: translateY(-1px);
        }

        div[data-testid="stButton"] > button:active {
            transform: translateY(0);
        }

        .tg-result-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 1.75rem 1.5rem;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
            border: 1px solid #e2ebe2;
            margin-top: 1.25rem;
        }

        .tg-result-label {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #7a8680;
            margin-bottom: 0.35rem;
        }

        .tg-result-name {
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0 0 0.25rem 0;
            line-height: 1.25;
        }

        .tg-result-name.healthy {
            color: #2e7d32;
        }

        .tg-result-name.disease {
            color: #e08a00;
        }

        .tg-result-key {
            font-size: 0.78rem;
            color: #9aa89e;
            margin-bottom: 1.25rem;
        }

        .tg-confidence-row {
            display: flex;
            align-items: center;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .tg-ring-wrap {
            flex-shrink: 0;
        }

        .tg-ring-label {
            font-size: 0.82rem;
            font-weight: 600;
            color: #5f6b62;
            margin-bottom: 0.35rem;
        }

        .tg-top3 {
            margin-top: 0.25rem;
        }

        .tg-top3-title {
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #7a8680;
            margin-bottom: 0.85rem;
        }

        .tg-bar-row {
            margin-bottom: 0.85rem;
        }

        .tg-bar-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 0.35rem;
            gap: 0.75rem;
        }

        .tg-bar-name {
            font-size: 0.88rem;
            color: #3d4a40;
            font-weight: 500;
        }

        .tg-bar-row:first-child .tg-bar-name {
            font-weight: 700;
            color: #1b3a1f;
        }

        .tg-bar-pct {
            font-size: 0.82rem;
            color: #5f6b62;
            font-weight: 600;
            white-space: nowrap;
        }

        .tg-bar-track {
            height: 8px;
            background: #e8f0e8;
            border-radius: 999px;
            overflow: hidden;
        }

        .tg-bar-row:first-child .tg-bar-track {
            height: 10px;
        }

        .tg-bar-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.4s ease;
        }

        .tg-bar-fill.primary {
            background: linear-gradient(90deg, #2e7d32, #66bb6a);
        }

        .tg-bar-fill.secondary {
            background: #b8d4ba;
        }

        .tg-info-box {
            background: #f0f7f0;
            border-left: 4px solid #66bb6a;
            border-radius: 0 12px 12px 0;
            padding: 1rem 1.1rem;
            margin: 1.25rem 0 1rem 0;
        }

        .tg-info-box-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: #2e7d32;
            margin-bottom: 0.4rem;
        }

        .tg-info-box p {
            margin: 0;
            font-size: 0.9rem;
            line-height: 1.6;
            color: #4a574c;
        }

        .tg-callout {
            border-radius: 12px;
            padding: 0.85rem 1rem;
            font-size: 0.9rem;
            font-weight: 600;
            line-height: 1.55;
        }

        .tg-callout.success {
            background: #e8f5e9;
            color: #2e7d32;
            border: 1px solid #a5d6a7;
        }

        .tg-callout.warning {
            background: #fff6e6;
            color: #e08a00;
            border: 1px solid #f5d38a;
        }

        @media (max-width: 640px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .tg-header h1 {
                font-size: 1.5rem;
            }

            .tg-confidence-row {
                flex-direction: column;
                align-items: center;
                text-align: center;
            }

            .tg-result-name {
                font-size: 1.45rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render centered app header with leaf icon and subtitle."""
    render_html(
        """
        <div class="tg-header">
            <div class="tg-header-icon">🌿</div>
            <h1>Multi-Crop Leaf Disease Classifier</h1>
            <p>Tomato case study — 10 disease &amp; healthy classes · EfficientNetB0</p>
        </div>
        <hr class="tg-divider" />
        """
    )


def render_empty_state() -> None:
    """Render hint card shown before any upload."""
    render_html(
        """
        <div class="tg-empty-state">
            <div class="tg-empty-state-icon">🌿</div>
            <p>Upload a leaf image to begin</p>
        </div>
        """
    )


def confidence_ring_svg(confidence: float, size: int = 112) -> str:
    """Build an inline SVG circular confidence ring."""
    pct = max(0.0, min(confidence * 100, 100.0))
    radius = 44
    circumference = 2 * 3.141592653589793 * radius
    dash = (pct / 100.0) * circumference
    gap = circumference - dash
    center = size // 2
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img"
         aria-label="Confidence {pct:.1f} percent">
        <circle cx="{center}" cy="{center}" r="{radius}"
                fill="none" stroke="#e8f0e8" stroke-width="10"/>
        <circle cx="{center}" cy="{center}" r="{radius}"
                fill="none" stroke="#2e7d32" stroke-width="10"
                stroke-linecap="round"
                stroke-dasharray="{dash:.2f} {gap:.2f}"
                transform="rotate(-90 {center} {center})"/>
        <text x="{center}" y="{center - 4}" text-anchor="middle"
              font-size="22" font-weight="700" fill="#1b3a1f">{pct:.0f}%</text>
        <text x="{center}" y="{center + 16}" text-anchor="middle"
              font-size="11" font-weight="500" fill="#7a8680">confidence</text>
    </svg>
    """


def render_top3_bars(top3: list[tuple[str, float]]) -> str:
    """Render top-3 prediction bars as HTML."""
    rows = []
    for rank, (label, prob) in enumerate(top3, start=1):
        pct = prob * 100
        fill_class = "primary" if rank == 1 else "secondary"
        rows.append(
            f"""
            <div class="tg-bar-row">
                <div class="tg-bar-header">
                    <span class="tg-bar-name">{html.escape(label)}</span>
                    <span class="tg-bar-pct">{pct:.1f}%</span>
                </div>
                <div class="tg-bar-track">
                    <div class="tg-bar-fill {fill_class}" style="width:{pct:.1f}%;"></div>
                </div>
            </div>
            """
        )
    return "\n".join(rows)


def render_result_card(
    top1_class: str,
    top1_pretty: str,
    top1_conf: float,
    top3: list[tuple[str, float]],
    description: str,
    is_healthy: bool,
) -> None:
    """Render styled prediction result card."""
    name_class = "healthy" if is_healthy else "disease"
    callout_class = "success" if is_healthy else "warning"
    if is_healthy:
        callout_text = "✓ This leaf appears healthy."
    else:
        callout_text = f"⚠ Possible {html.escape(top1_pretty.lower())} detected."

    render_html(
        f"""
        <div class="tg-result-card">
            <div class="tg-result-label">Prediction</div>
            <p class="tg-result-name {name_class}">{html.escape(top1_pretty)}</p>
            <div class="tg-result-key">Class key: {html.escape(top1_class)}</div>
            <div class="tg-confidence-row">
                <div class="tg-ring-wrap">
                    <div class="tg-ring-label">Model confidence</div>
                    {confidence_ring_svg(top1_conf)}
                </div>
            </div>
            <div class="tg-top3">
                <div class="tg-top3-title">Top 3 predictions</div>
                {render_top3_bars(top3)}
            </div>
            <div class="tg-info-box">
                <div class="tg-info-box-title">About this condition</div>
                <p>{html.escape(description)}</p>
            </div>
            <div class="tg-callout {callout_class}">
                {callout_text}
            </div>
        </div>
        """
    )


def render_sidebar(label_map: dict, img_size: tuple[int, int]) -> None:
    """Render sidebar with model info, about text, and disclaimer."""
    model_name = html.escape(str(label_map.get("best_model", "EfficientNetB0")))
    num_classes = label_map.get("num_classes", 10)
    accuracy_pct = REPORTED_TEST_ACCURACY * 100

    with st.sidebar:
        st.markdown(
            f"""
            <div class="tg-sidebar-panel">
                <h3>Model Information</h3>
                <div class="tg-sidebar-meta">
                    <span><strong>Model:</strong> {model_name}</span>
                    <span><strong>Input size:</strong> {img_size[0]} × {img_size[1]}</span>
                    <span><strong>Classes:</strong> {num_classes}</span>
                    <span><strong>Reported test accuracy:</strong> ~{accuracy_pct:.1f}%
                    (from the project's Kaggle evaluation)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="tg-sidebar-panel">
                <h3>About</h3>
                <p>TomatoGuard-CNN is a student deep-learning project that classifies
                tomato leaf images into 10 disease and healthy categories using
                transfer learning.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="tg-sidebar-panel">
                <h3>Disclaimer</h3>
                <p class="tg-disclaimer">{html.escape(DISCLAIMER)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Multi-Crop Leaf Disease Classifier",
        page_icon="🌿",
        layout="centered",
    )

    inject_css()

    model = load_model()
    label_map = load_label_map()

    if model is None or label_map is None:
        st.error(
            "Required files not found. Please place `best_model.keras` and "
            "`label_map.json` in the app folder (repo root), or in `results/`."
        )
        st.stop()

    img_size_list = label_map.get("img_size", [224, 224])
    img_size = (int(img_size_list[0]), int(img_size_list[1]))
    index_to_class = build_index_to_class(label_map)

    render_sidebar(label_map, img_size)
    render_header()

    uploaded = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )

    if uploaded is None:
        render_empty_state()
    else:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded leaf", width=300)

        if st.button("Predict", type="primary"):
            try:
                with st.spinner("Analyzing leaf..."):
                    x = preprocess_image(image, img_size)
                    preds = model.predict(x, verbose=0)[0]

                top_indices = np.argsort(preds)[::-1]
                top1_idx = int(top_indices[0])
                top1_class = lookup_class(index_to_class, top1_idx)
                top1_conf = float(preds[top1_idx])
                top1_pretty = display_name(top1_class)

                top3: list[tuple[str, float]] = []
                for idx in top_indices[:3]:
                    idx_int = int(idx)
                    class_key = lookup_class(index_to_class, idx_int)
                    prob = float(preds[idx_int])
                    top3.append((display_name(class_key), prob))

                description = DISEASE_INFO.get(
                    top1_class,
                    "No description available for this class.",
                )

                render_result_card(
                    top1_class=top1_class,
                    top1_pretty=top1_pretty,
                    top1_conf=top1_conf,
                    top3=top3,
                    description=description,
                    is_healthy="healthy" in top1_class.lower(),
                )

            except Exception as exc:
                st.error("Something went wrong during prediction. Please try another image.")
                st.error(f"{type(exc).__name__}: {exc}")
                st.exception(exc)


if __name__ == "__main__":
    main()
