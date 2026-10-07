import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import os
import io
from datetime import datetime
import tempfile

# --------------------------- Helper functions ---------------------------

def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)


def pil_to_cv2(img_pil: Image.Image):
    """Convert PIL Image to OpenCV BGR format"""
    img = np.array(img_pil.convert('RGB'))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def cv2_to_pil(img_cv2):
    """Convert OpenCV BGR/gray array to PIL Image"""
    if len(img_cv2.shape) == 2:
        return Image.fromarray(img_cv2)
    rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_for_display(img_cv2, max_dim=1024):
    h, w = img_cv2.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img_cv2 = cv2.resize(img_cv2, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img_cv2


def remove_background_grabcut(img_bgr, rect=None, iter_count=5):
    """Simple background removal using GrabCut."""
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    h, w = img_bgr.shape[:2]
    if rect is None:
        margin_w = int(w * 0.07)
        margin_h = int(h * 0.07)
        rect = (margin_w, margin_h, w - 2 * margin_w, h - 2 * margin_h)
    cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_fg = img_bgr * mask2[:, :, np.newaxis]
    white_bg = np.ones_like(img_bgr, dtype=np.uint8) * 255
    result = np.where(mask2[:, :, np.newaxis] == 1, img_fg, white_bg)
    return result


def adjust_contrast_brightness(img_gray, contrast=1.0, brightness=0):
    result = cv2.convertScaleAbs(img_gray, alpha=contrast, beta=brightness)
    return result


def pencil_sketch(img_bgr, blur_ksize=21, contrast=1.0, brightness=0, edge_strength=1.0):
    img_bgr = resize_for_display(img_bgr, max_dim=1200)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = adjust_contrast_brightness(gray, contrast, brightness)
    inv = cv2.bitwise_not(gray)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blur = cv2.GaussianBlur(inv, (blur_ksize, blur_ksize), 0)
    inv_blur = cv2.bitwise_not(blur)
    sketch = cv2.divide(gray, inv_blur, scale=256.0)
    if edge_strength != 1.0:
        sketch = cv2.convertScaleAbs(sketch, alpha=edge_strength, beta=0)
    return sketch


def soft_sketch(img_bgr, blur_ksize=35, contrast=1.1, brightness=5):
    gray = cv2.cvtColor(resize_for_display(img_bgr), cv2.COLOR_BGR2GRAY)
    gray = adjust_contrast_brightness(gray, contrast, brightness)
    inv = 255 - gray
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blur = cv2.GaussianBlur(inv, (blur_ksize, blur_ksize), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    sketch = cv2.medianBlur(sketch, 3)
    return sketch


def cartoonify(img_bgr, downscale=2, num_bilateral=7, edge=5):
    img = resize_for_display(img_bgr)
    img_color = img.copy()
    for _ in range(int(np.log2(downscale)) if downscale > 1 else 0):
        img_color = cv2.pyrDown(img_color)
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
    for _ in range(int(np.log2(downscale)) if downscale > 1 else 0):
        img_color = cv2.pyrUp(img_color)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    edges = cv2.adaptiveThreshold(img_blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=9, C=2)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(img_color, edges_colored)
    if edge != 1:
        cartoon = cv2.convertScaleAbs(cartoon, alpha=edge, beta=0)
    return cartoon


def edge_only(img_bgr, threshold1=50, threshold2=150):
    img = resize_for_display(img_bgr)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return edges


def save_image_pil(img_pil, prefix="sketch"):
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{ts}.png"
    path = os.path.join("outputs", filename)
    img_pil.save(path)
    return path


# --------------------------- Streamlit UI ---------------------------

st.set_page_config(page_title="Pro Image to Sketch", layout="wide")

st.markdown("""
# ✨ Professional Image to Sketch
Upload an image or use your camera. Choose a sketch style, tune parameters, preview and download.
""")

# Sidebar controls
with st.sidebar:
    st.header("🛠️ Controls")
    source = st.radio("Image source:", ("Upload", "Camera"))
    style = st.selectbox("Sketch Style:", ("Pencil", "Soft Sketch", "Cartoon", "Edge Only"))
    st.markdown("---")
    st.write("### Adjustments")
    blur_val = st.slider("Blur kernel (affects sketch smoothness)", 1, 99, 21, step=2)
    edge_strength = st.slider("Edge Strength / Intensity", 0.5, 3.0, 1.0, step=0.1)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, step=0.05)
    brightness = st.slider("Brightness", -50, 50, 0, step=1)
    remove_bg = st.checkbox("Remove background (GrabCut)", value=False)
    if remove_bg:
        st.caption("Background removal works best with clear foreground and simple backgrounds.")
    st.markdown("---")
    st.write("### Export")
    auto_save = st.checkbox("Auto save results to outputs/", value=True)
    st.write("Outputs will be saved with timestamped filenames.")

# Main layout: two columns for preview
col1, col2 = st.columns([1, 1])

uploaded_image = None

if source == "Upload":
    uploaded_file = st.file_uploader("Upload image (jpg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            uploaded_image = image.copy()
        except Exception as e:
            st.error(f"Could not open the image: {e}")

else:
    cam_img = st.camera_input("Take a picture")
    if cam_img is not None:
        try:
            image = Image.open(cam_img)
            uploaded_image = image.copy()
        except Exception as e:
            st.error(f"Could not read camera image: {e}")

if uploaded_image is not None:
    img_bgr = pil_to_cv2(uploaded_image)

    if remove_bg:
        with st.spinner("Removing background..."):
            try:
                img_bgr = remove_background_grabcut(img_bgr)
            except Exception as e:
                st.warning("Background removal failed; using original image.")

    # Generate result based on style
    result_img = None
    with st.spinner("Generating sketch..."):
        if style == "Pencil":
            result_img = pencil_sketch(img_bgr, blur_ksize=blur_val, contrast=contrast, brightness=brightness, edge_strength=edge_strength)
            result_pil = cv2_to_pil(result_img)
        elif style == "Soft Sketch":
            result_img = soft_sketch(img_bgr, blur_ksize=blur_val, contrast=contrast, brightness=brightness)
            result_pil = cv2_to_pil(result_img)
        elif style == "Cartoon":
            result_img = cartoonify(img_bgr, downscale=2, num_bilateral=5, edge=edge_strength)
            result_pil = cv2_to_pil(result_img)
        else:  # Edge Only
            result_img = edge_only(img_bgr, threshold1=50, threshold2=150)
            result_pil = cv2_to_pil(result_img)

    # Show images side-by-side
    with col1:
        st.subheader("Original")
        st.image(uploaded_image, use_column_width=True)

    with col2:
        st.subheader("Result")
        st.image(result_pil, use_column_width=True)

    # Download and save
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.download_button("⬇️ Download Result", data=byte_im, file_name="sketch.png", mime="image/png")
    with c2:
        st.download_button("⬇️ Download Original", data=uploaded_file.getvalue() if source == 'Upload' else None, file_name="original.png", mime="image/png")
    with c3:
        
     if auto_save:
        saved_path = save_image_pil(result_pil, prefix=style.replace(' ', '_').lower())
        st.info(f"Auto-saved result: {saved_path}")

else:
    st.info("Upload an image or use the camera to get started.")

# Footer
st.markdown("---")
st.markdown(" Built with Streamlit + OpenCV")
