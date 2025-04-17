import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi, threshold_local
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage import exposure, morphology, filters, segmentation
import os
import tempfile
from io import BytesIO

st.set_page_config(
    page_title="Fundus Image Hemorrhage Detection",
    layout="wide"
)

st.title("Fundus Image Hemorrhage Detection")
st.write("Upload a fundus image to detect hemorrhages and damaged blood vessels using two different methods.")

# File uploader
uploaded_file = st.file_uploader("Choose a fundus image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Fundus Image', use_column_width=True)
    
    # Add a "Process" button
    if st.button("Process Image"):
        st.write("Processing...")
        
        # Method selection
        method_tab1, method_tab2 = st.tabs(["Method 1: Vessel + Hemorrhage Detection", "Method 2: Step-wise Hemorrhage Detection"])
        
        # Method 1: Vessel damage detection
        with method_tab1:
            # Load the fundus image
            image = cv2.imread(temp_file_path)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Frangi filter to enhance blood vessels
            vessel_enhanced = frangi(gray)
            
            # Convert to binary using thresholding
            _, vessel_binary = cv2.threshold((vessel_enhanced * 255).astype(np.uint8), 20, 255, cv2.THRESH_BINARY)
            
            # Remove small noise objects
            vessel_binary = remove_small_objects(vessel_binary.astype(bool), min_size=100).astype(np.uint8) * 255
            
            # Convert image to HSV and extract red channel for hemorrhages
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            red_mask1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))  # Detect red lesions
            red_mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))  
            red_areas = red_mask1 | red_mask2  # Combine both red ranges
            
            # Detect damaged blood vessel areas by finding overlap between vessels and red hemorrhages
            damaged_areas = cv2.bitwise_and(vessel_binary, red_areas)
            
            # Overlay detected damaged areas on the original image
            highlighted_damage = image.copy()
            highlighted_damage[damaged_areas == 255] = [0, 0, 255]  # Highlight damage in red
            
            # Display results
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original Fundus Image")
            axes[1].imshow(red_areas, cmap='gray')
            axes[1].set_title("Detected Hemorrhage Areas")
            axes[2].imshow(cv2.cvtColor(highlighted_damage, cv2.COLOR_BGR2RGB))
            axes[2].set_title("Blood Vessel Leakage Highlighted (Red)")
            
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            
            # Save figure to buffer
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            
            # Display in Streamlit
            st.image(buf, use_column_width=True)
            plt.close(fig)
        
        # Method 2: Step-wise detection
        with method_tab2:
            # Read the image
            img = cv2.imread(temp_file_path)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize and extract Green channel
            resized = cv2.resize(img, (512, 512))  # Standardize size
            green = resized[:, :, 1]  # Green channel (index 1 in BGR)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_green = clahe.apply(green)
            
            # Complement the CLAHE image
            complement = 255 - clahe_green
            
            # Morphological Opening & Subtraction
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened = cv2.morphologyEx(complement, cv2.MORPH_OPEN, kernel)
            subtracted = cv2.subtract(complement, opened)
            
            # Median Filtering & Subtraction
            median = cv2.medianBlur(subtracted, 5)
            final_sub = cv2.subtract(median, opened)
            
            # Adjust Image Intensity
            adjusted = exposure.rescale_intensity(final_sub, in_range=(50, 200))
            
            # Complement Again
            final_complement = 255 - adjusted
            
            # Region Growing Segmentation (Local Thresholding)
            thresh = filters.threshold_local(final_complement, block_size=51, offset=10)
            binary = final_complement > thresh
            
            # Morphological Closing to Smooth Edges
            closed = morphology.binary_closing(binary, footprint=morphology.disk(3))
            
            # Create figure for display
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            axes[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original Image")
            
            axes[1].imshow(clahe_green, cmap='gray')
            axes[1].set_title("CLAHE Green Channel")
            
            axes[2].imshow(complement, cmap='gray')
            axes[2].set_title("Complemented CLAHE")
            
            axes[3].imshow(adjusted, cmap='gray')
            axes[3].set_title("Adjusted Intensity")
            
            axes[4].imshow(binary, cmap='gray')
            axes[4].set_title("Region Growing (Binary)")
            
            axes[5].imshow(closed, cmap='gray')
            axes[5].set_title("Final Hemorrhage Detection")
            
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            
            # Save figure to buffer
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            
            # Display in Streamlit
            st.image(buf, use_column_width=True)
            plt.close(fig)
    
    # Clean up the temp file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

st.write("---")
st.write("""
## About this app
This application uses two different methods to detect hemorrhages in fundus images:

1. **Method 1**: Combines blood vessel detection using the Frangi filter with hemorrhage detection based on HSV color space analysis.
2. **Method 2**: Uses a step-wise approach with green channel extraction, CLAHE enhancement, morphological operations, and region growing segmentation.

Upload a fundus image to see the results from both methods.
""")
