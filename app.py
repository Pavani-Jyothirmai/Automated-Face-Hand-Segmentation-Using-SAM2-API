import os
import base64
import cv2
import requests
import numpy as np
import replicate
import mediapipe as mp
import streamlit as st
from PIL import Image
import time



# ---------------- Setup -------------------
replicate_token = os.getenv("REPLICATE_API_TOKEN")
if replicate_token is None:
    raise ValueError("⚠️ Please set the REPLICATE_API_TOKEN environment variable.")
os.environ["REPLICATE_API_TOKEN"] = replicate_token


# Retry-safe GET with timeout
def safe_get(url, retries=3, delay=3):
    for attempt in range(retries):
        try:
            return requests.get(url, timeout=(30, 180))
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e



# ---------------- Utility: Expand bounding box -------------------
def expand_box(box, img_w, img_h, scale=1.6):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bw = int((x2 - x1) * scale)
    bh = int((y2 - y1) * scale)
    new_x1 = max(cx - bw // 2, 0)
    new_y1 = max(cy - bh // 2, 0)
    new_x2 = min(cx + bw // 2, img_w)
    new_y2 = min(cy + bh // 2, img_h)
    return [new_x1, new_y1, new_x2, new_y2]

# ---------------- Step 1: Detect face & hands -------------------
def detect_face_hand_boxes(image):
    mp_face = mp.solutions.face_detection
    mp_hands = mp.solutions.hands
    h, w, _ = image.shape
    boxes, labels = [], []

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        result = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if result.detections:
            for det in result.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w)
                y2 = y1 + int(bbox.height * h)
                boxes.append([x1, y1, x2, y2])
                labels.append("face")

    with mp_hands.Hands(static_image_mode=True, max_num_hands=4, min_detection_confidence=0.5) as hands:
        result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            for i, lm in enumerate(result.multi_hand_landmarks):
                xs = [pt.x for pt in lm.landmark]
                ys = [pt.y for pt in lm.landmark]
                x1 = int(min(xs) * w)
                y1 = int(min(ys) * h)
                x2 = int(max(xs) * w)
                y2 = int(max(ys) * h)
                boxes.append([x1, y1, x2, y2])
                labels.append(f"hand-{i+1}")

    return [expand_box(b, w, h) for b in boxes], labels

# ---------------- Step 2: Get best mask -------------------
def get_best_mask_url(box, image_shape, masks):
    x1, y1, x2, y2 = box
    h, w = image_shape
    max_iou, best_url = 0, None

    for m in masks:
        try:
            url = m.url
            response = safe_get(url)
            mask = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)

            if mask is None:
                continue
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]

            mask = cv2.resize(mask, (w, h))
            bin_mask = (mask > 128).astype(np.uint8)

            box_mask = np.zeros_like(bin_mask)
            box_mask[y1:y2, x1:x2] = 1

            intersection = np.logical_and(bin_mask, box_mask).sum()
            union = np.logical_or(bin_mask, box_mask).sum()
            iou = intersection / union if union else 0

            if iou > max_iou:
                max_iou, best_url = iou, url
        except:
            continue

    return best_url


# ---------------- Step 3: Draw masks & boxes -------------------
def overlay_masks_and_boxes(image, boxes, mask_urls, labels):
    vis = image.copy()
    for url in mask_urls:
        mask = cv2.imdecode(np.frombuffer(requests.get(url).content, np.uint8), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        bin_mask = (mask > 128).astype(np.uint8)
        color = np.full_like(vis, (138, 43, 226))
        vis = np.where(bin_mask[:, :, None], cv2.addWeighted(vis, 0.6, color, 0.4, 0), vis)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        label = labels[i]
        text_y = y1 + 20
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(vis, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(vis, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis

# ---------------- Streamlit UI -------------------
st.set_page_config(page_title="Face & Hand Segmentation using SAM2", layout="centered")

st.markdown("""
<h1 style='text-align: center;'>👤✋ Face & Hand Segmenter with SAM2</h1>
<h4 style='text-align: center; color: violet;'>"📁 Upload an Image and watch the magic! ✨</h4>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize for performance
    max_width = 256
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(image, (max_width, int(image.shape[0] * scale)))


    with st.spinner("🔍 Processing..."):
        try:
            boxes, labels = detect_face_hand_boxes(image)
            h, w, _ = image.shape
            _, img_encoded = cv2.imencode(".jpg", image)
            b64img = base64.b64encode(img_encoded.tobytes()).decode()
            data_url = f"data:image/jpeg;base64,{b64img}"

            mask_urls = []
            for box in boxes:
                x1, y1, x2, y2 = box
                input_box = [[x1, y1, x2 - x1, y2 - y1]]

                try:
                    output = replicate.run(
                        "lucataco/segment-anything-2:be7cbde9fdf0eecdc8b20ffec9dd0d1cfeace0832d4d0b58a071d993182e1be0",
                    input={"image": data_url, "input_boxes": input_box}
                    )
                    best_url = get_best_mask_url(box, (h, w), output)
                    if best_url:
                        mask_urls.append(best_url)

                except replicate.exceptions.ReplicateError as e:
                    if "429" in str(e):
                        st.warning("⚠️ Rate limit hit. Waiting 5 seconds before retrying...")
                        time.sleep(5)  # wait before retrying
                        continue
                    else:
                        raise e

                time.sleep(5)

            result_image = overlay_masks_and_boxes(image, boxes, mask_urls, labels)

            # Side-by-side display
            st.subheader("🖼 Original vs 🎯 Segmented")
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
            with col2:
                st.image(result_image, caption="Segmented Output", use_container_width=True)

            st.download_button(
                label="📥 Download Segmented Image",
                data=cv2.imencode(".jpg", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))[1].tobytes(),
                file_name="segmented_output.jpg",
                mime="image/jpeg"
            )

        except Exception as e:
            st.error(f"❌ Failed to process image: {e}")
