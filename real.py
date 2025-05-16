import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from scipy.ndimage import maximum_filter, label, gaussian_filter
from PIL import Image

# CSRNet Model Definition
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.backend = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

# Head detection
def detect_heads(density_map, threshold=0.2):
    smoothed = gaussian_filter(density_map, sigma=2)
    filtered = maximum_filter(smoothed, size=15)
    peaks = (smoothed == filtered) & (smoothed > threshold)
    labeled, num_heads = label(peaks)
    return num_heads, labeled

@st.cache_resource(show_spinner=False)
def load_model():
    model = CSRNet()
    model.load_state_dict(torch.load("csrnet.pth", map_location="cpu"))
    model.eval()
    return model

def main():
    st.title("Real-time Crowd Counting with CSRNet")

    # Load model once
    model = load_model()
    device = torch.device("cpu")
    model.to(device)

    # Webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    stframe = st.empty()
    count_text = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame")
            break

        # Convert frame and predict
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
        density_map = output.squeeze().cpu().numpy()

        # Detect heads
        count, labeled_heads = detect_heads(density_map)

        # Draw boxes on original frame
        for label_num in range(1, count + 1):
            ys, xs = np.where(labeled_heads == label_num)
            if len(xs) > 0 and len(ys) > 0:
                x, y = int(xs.mean()), int(ys.mean())
                h, w, _ = frame.shape
                x = int(x * w / 512)
                y = int(y * h / 512)
                cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (255, 0, 0), 2)
                cv2.putText(frame, "H", (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.putText(frame, f"Crowd Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert BGR to RGB for displaying in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        # Display frame and count
        stframe.image(img_pil, channels="RGB")
        count_text.markdown(f"### Crowd Count: {count}")

        # Streamlit’s limitation: To break the loop gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == "__main__":
    main()
