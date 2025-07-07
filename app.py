import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from streamlit_drawable_canvas import st_canvas
import os, io, zipfile, requests
from datetime import datetime
from scipy.optimize import curve_fit
import json
from dotenv import load_dotenv 
load_dotenv()
# Set Streamlit page configuration for a wide layout and a custom title.
st.set_page_config(page_title="ThermalScope: Advanced DED Melt Pool Analysis", layout="wide")

# --- Constants and Configuration ---
# Define color palette for Streamlit elements to enhance UX.
PRIMARY_COLOR = "#0E1117"
SECONDARY_BACKGROUND_COLOR = "#262730"
TEXT_COLOR = "#FAFAFA"
ACCENT_COLOR = "#FF4B4B" # Streamlit's default red for buttons/highlights

# --- Session State Initialization ---
# Initialize session state variables to store computed data and UI states, preventing re-computation.
if 'roi_data' not in st.session_state:
    st.session_state.roi_data = {} # Stores analysis results for each ROI
if 'current_image_key' not in st.session_state:
    st.session_state.current_image_key = None
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = {} # Stores original images
if 'compare_mode' not in st.session_state:
    st.session_state.compare_mode = False
if 'selected_rois_to_compare' not in st.session_state:
    st.session_state.selected_rois_to_compare = []

# --- Helper Functions for Data Handling and UI ---

def get_unique_key(prefix="analysis"):
    """Generates a unique key based on current timestamp for session state management."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

@st.cache_data
def convert_df_to_csv(df):
    """Converts a Pandas DataFrame to CSV format for download."""
    return df.to_csv(index=False).encode('utf-8')

def download_data_button(df, name):
    """Creates a Streamlit download button for a given DataFrame."""
    st.download_button(
        label=f"Download {name}.csv",
        data=convert_df_to_csv(df),
        file_name=f"{name}.csv",
        mime="text/csv",
        key=f"download_{name}_{get_unique_key('csv')}"
    )

def create_zip_archive(data_dict):
    """
    Creates a ZIP archive from a dictionary of DataFrames.
    Args:
        data_dict (dict): A dictionary where keys are filenames (without .csv)
                          and values are Pandas DataFrames.
    Returns:
        bytes: The content of the ZIP file as bytes.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, df in data_dict.items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False))
    return buffer.getvalue()

def download_zip_button(data_dict, zip_name="thermalscope_data"):
    """Creates a Streamlit download button for a ZIP archive of DataFrames."""
    zip_data_bytes = create_zip_archive(data_dict)
    st.download_button(
        label="Download All Data (ZIP)",
        data=zip_data_bytes,
        file_name=f"{zip_name}.zip",
        mime="application/zip",
        key=f"download_{zip_name}_{get_unique_key('zip')}"
    )

# --- Image Processing Utilities ---

def enhance_image(img, clip_limit=3.0, tile_grid_size=8):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.

    Args:
        img (np.array): The input image (RGB or BGR).
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (int): Size of grid for histogram equalization. Input image is divided into
                              this many regions.

    Returns:
        np.array: The contrast-enhanced image in RGB format.
    """
    # Ensure image is in BGR for OpenCV processing, then convert to LAB.
    # Convert RGB to BGR first if the input is RGB from PIL.
    if img.shape[-1] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img # Assuming grayscale or already BGR if not 3 channels.

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    # Convert back to RGB for Streamlit display.
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def extract_thermal_profile(img):
    """
    Extracts the mean thermal intensity profile across the width of the image.

    Args:
        img (np.array): The input image (RGB or grayscale).

    Returns:
        np.array: A 1D array representing the mean thermal profile.
    """
    # Convert to grayscale if the image is RGB.
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img # Already grayscale
    return np.mean(gray, axis=0) # Mean intensity along the vertical axis (across columns)

def compute_cooling_rate(profile):
    """
    Computes the cooling rate from a thermal profile using numerical gradient.

    Args:
        profile (np.array): A 1D array of thermal intensities.

    Returns:
        np.array: A 1D array representing the cooling rate (gradient of the profile).
    """
    return np.gradient(profile)

def compute_gradient_map(img):
    """
    Computes the absolute intensity gradient map of an image, typically along the scan direction.

    Args:
        img (np.array): The input image (RGB or grayscale).

    Returns:
        np.array: A 2D array representing the absolute intensity gradient map.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
    else:
        gray = img.astype(float)
    # Compute gradient along the horizontal axis (x-direction, axis=1) for melt pool elongation.
    return np.abs(np.gradient(gray, axis=1))

def compute_GR_ratio(grad_map, solidification_rate):
    """
    Calculates the G/R ratio (Temperature Gradient / Solidification Rate).

    Args:
        grad_map (np.array): The intensity gradient map.
        solidification_rate (float): The known or estimated solidification rate.

    Returns:
        float: The computed G/R ratio. Returns 0 if solidification_rate is zero.
    """
    G = np.mean(grad_map) # Average gradient across the map
    return G / solidification_rate if solidification_rate != 0 else 0

# --- Scientific Inference Modules ---

def predict_microstructure(GR_ratio):
    """
    Predicts the microstructure based on the G/R ratio.

    Args:
        GR_ratio (float): The G/R ratio.

    Returns:
        str: The predicted microstructure type (Equiaxed, Mixed, or Columnar).
    """
    if GR_ratio < 0.5:
        return "Equiaxed"
    elif GR_ratio < 2.5:
        return "Mixed"
    else:
        return "Columnar"

def estimate_HAZ_width(img, threshold_factor=0.3):
    """
    Estimates the Heat Affected Zone (HAZ) width based on a thermal intensity drop.
    The HAZ is estimated by finding regions where the intensity drops significantly from the peak.

    Args:
        img (np.array): The input image (RGB or grayscale).
        threshold_factor (float): Factor of the peak intensity to determine HAZ boundaries.

    Returns:
        int: The estimated HAZ width in pixels. Returns 0 if no clear HAZ is found.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Consider a central vertical strip to estimate the HAZ.
    center_strip = np.mean(gray[:, img.shape[1]//2-5:img.shape[1]//2+5], axis=1)
    peak_intensity = np.max(center_strip)
    
    # Find points where intensity drops below a certain threshold.
    # We look for where the intensity drops significantly, indicative of the HAZ edge.
    # Search from both ends towards the peak.
    threshold_value = threshold_factor * peak_intensity
    
    # Find first index from start where value is below threshold
    start_boundary_idx = -1
    for i in range(len(center_strip)):
        if center_strip[i] < threshold_value:
            start_boundary_idx = i
            break
            
    # Find first index from end where value is below threshold
    end_boundary_idx = -1
    for i in range(len(center_strip) - 1, -1, -1):
        if center_strip[i] < threshold_value:
            end_boundary_idx = i
            break

    if start_boundary_idx != -1 and end_boundary_idx != -1 and start_boundary_idx < end_boundary_idx:
        return end_boundary_idx - start_boundary_idx
    return 0

def fit_2d_gaussian(img):
    """
    Fits a 2D Gaussian function to the image intensity data to model the melt pool shape.

    Args:
        img (np.array): The input image (RGB or grayscale).

    Returns:
        tuple: A dictionary of Gaussian parameters (Amplitude, X Center, Y Center, Sigma X, Sigma Y, Offset)
               and a tuple of the raw fitted parameters.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    h, w = gray.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    data = gray.flatten()

    def gauss2d(coords, amp, xo, yo, sx, sy, offset):
        """2D Gaussian function for curve fitting."""
        x_coords, y_coords = coords
        return amp * np.exp(-((x_coords - xo)**2 / (2*sx**2) + (y_coords - yo)**2 / (2*sy**2))) + offset

    # Initial guess for parameters: amplitude (max intensity), center (image center),
    # sigma (quarter of width/height), and offset (min intensity).
    p0 = [np.max(gray), w // 2, h // 2, w / 4, h / 4, np.min(gray)]
    
    try:
        popt, _ = curve_fit(gauss2d, (x.flatten(), y.flatten()), data, p0=p0)
    except RuntimeError:
        # If curve_fit fails (e.g., singular matrix), return default/invalid values.
        st.warning("Gaussian fit failed to converge. Using default parameters.")
        popt = p0 # Fallback to initial guess if fit fails.

    keys = ["Amplitude", "X Center", "Y Center", "Sigma X", "Sigma Y", "Offset"]
    return dict(zip(keys, popt)), popt

def compute_shape_residual_map(img, gaussian_params):
    """
    Computes the residual map by subtracting the 2D Gaussian fit from the original image.
    This highlights deviations from the ideal Gaussian melt pool shape.

    Args:
        img (np.array): The input image (RGB or grayscale).
        gaussian_params (tuple): The fitted 2D Gaussian parameters (amp, xo, yo, sx, sy, offset).

    Returns:
        np.array: A 2D array representing the residual (difference) map.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    h, w = gray.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    amp, xo, yo, sx, sy, offset = gaussian_params
    
    # Reconstruct the 2D Gaussian model. Handle potential division by zero for sx, sy.
    model = np.zeros_like(gray, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'): # Suppress warnings for division by zero
        model = amp * np.exp(-((x - xo)**2 / (2*sx**2 if sx != 0 else 1) + 
                                (y - yo)**2 / (2*sy**2 if sy != 0 else 1))) + offset
    
    # Ensure model is within valid intensity range if it goes out due to fitting.
    model = np.clip(model, 0, 255)
    
    return gray.astype(float) - model

def extract_meltpool_geometry_stats(img):
    """
    Extracts geometric statistics (Area, Centroid, Bounding Box) of melt pool regions.

    Args:
        img (np.array): The input image (RGB or grayscale), typically an enhanced ROI.

    Returns:
        list: A list of dictionaries, each containing stats for a detected melt pool contour.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Apply Otsu's thresholding to binarize the image and isolate melt pool regions.
    # Morphological operations (e.g., closing) can be added here for better contour detection.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours in the thresholded image.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stats = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter small contours that might be noise.
        if area > 10: # Minimum area threshold for a valid melt pool region.
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00']) # Centroid X
                cy = int(M['m01'] / M['m00']) # Centroid Y
                x, y, w, h = cv2.boundingRect(cnt) # Bounding Box
                stats.append({
                    "Area": area,
                    "Centroid X": cx,
                    "Centroid Y": cy,
                    "BBox Width": w,
                    "BBox Height": h
                })
    return stats

def compute_crack_risk_index(GR_ratio, gradient_map, cooling_rate):
    """
    Estimates a Crack Risk Index (CRI) based on G/R ratio, gradient peak zones, and cooling rate.
    This is a mock formula for demonstration and can be refined based on scientific models.

    Args:
        GR_ratio (float): The G/R ratio.
        gradient_map (np.array): The intensity gradient map.
        cooling_rate (np.array): The 1D cooling rate profile.

    Returns:
        float: The computed Crack Risk Index. Higher values indicate higher risk.
    """
    # Simplified mock formula:
    # High G/R (columnar growth) generally increases risk.
    # High gradient standard deviation implies uneven solidification fronts.
    # Very high or very low cooling rates can induce stress.
    
    # Factor 1: G/R ratio contribution
    # Assuming higher G/R (towards columnar) means higher risk
    gr_risk_factor = GR_ratio * 0.5 

    # Factor 2: Gradient map uniformity/peak analysis
    # High standard deviation of gradient map implies non-uniform thermal gradients
    # which can lead to stress concentrations.
    gradient_std = np.std(gradient_map)
    gradient_risk_factor = gradient_std * 0.1

    # Factor 3: Cooling rate extremes
    # Very fast or very slow cooling can both be detrimental.
    # Let's consider deviation from an "ideal" cooling rate (e.g., median or typical rate).
    # For a simple mock, we'll consider the absolute peak cooling rate (most rapid cooling).
    peak_cooling_abs = np.abs(np.min(cooling_rate)) # Using min as cooling rate is negative typically
    cooling_risk_factor = peak_cooling_abs * 0.05
    
    # Combine factors. Weights can be adjusted based on expert knowledge.
    crack_risk = gr_risk_factor + gradient_risk_factor + cooling_risk_factor
    
    # Normalize or cap the index if needed. For now, just return the raw sum.
    return crack_risk

def generate_scientific_summary(GR, microstructure, cooling_profile, gauss_dict, haz_width, crack_risk):
    """
    Generates a comprehensive scientific summary of the melt pool analysis.

    Args:
        GR (float): The G/R ratio.
        microstructure (str): Predicted microstructure.
        cooling_profile (np.array): The 1D cooling rate profile.
        gauss_dict (dict): Dictionary of 2D Gaussian fit parameters.
        haz_width (int): Estimated HAZ width.
        crack_risk (float): Computed crack risk index.

    Returns:
        str: A formatted string containing the scientific summary.
    """
    peak_cooling_rate = np.min(cooling_profile) if cooling_profile.size > 0 else 0
    
    summary_parts = [
        f"**Thermal-Microstructure Analysis Report**\n",
        f"---",
        f"**1. G/R Ratio & Microstructure:**",
        f"   - **G/R Ratio:** {GR:.3f}",
        f"   - **Predicted Microstructure:** {microstructure} Growth",
        f"     (Based on G/R ratio thresholds: <0.5 Equiaxed, 0.5-2.5 Mixed, >2.5 Columnar)\n",
        f"**2. Thermal Characteristics:**",
        f"   - **Peak Cooling Rate:** {peak_cooling_rate:.3f} (units/pixel, indicative of rapid solidification)",
        f"   - **Heat Affected Zone (HAZ) Width:** {haz_width} pixels (estimated by {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n"
    ]

    if gauss_dict:
        sigma_x = gauss_dict.get("Sigma X", 0)
        sigma_y = gauss_dict.get("Sigma Y", 0)
        dominance = 'X-dominant' if sigma_x > sigma_y else 'Y-dominant'
        summary_parts.extend([
            f"**3. Melt Pool Geometry (Gaussian Fit):**",
            f"   - **Amplitude:** {gauss_dict.get('Amplitude', 0):.2f}",
            f"   - **Center (X, Y):** ({gauss_dict.get('X Center', 0):.1f}, {gauss_dict.get('Y Center', 0):.1f})",
            f"   - **Sigma (X, Y):** ({sigma_x:.1f}, {sigma_y:.1f})",
            f"   - **Thermal Distribution:** {dominance} (indicating melt pool elongation/spread along dominant axis)\n"
        ])
    
    if crack_risk is not None:
        summary_parts.append(f"**4. Process Risk Assessment:**")
        summary_parts.append(f"   - **Estimated Crack Risk Index (CRI):** {crack_risk:.3f}")
        summary_parts.append(f"     (Higher values indicate increased risk of cracking, based on G/R, gradient uniformity, and cooling rate extremes)\n")
    
    summary_parts.append(f"---")
    summary_parts.append(f"**Notes:**")
    summary_parts.append(f"   - All spatial measurements are in pixels relative to the ROI.")
    summary_parts.append(f"   - Thermal intensities are relative to the raw image values.")
    summary_parts.append(f"   - This report is generated by ThermalScope, a scientific analysis tool.")

    return "\n".join(summary_parts)

# --- Streamlit UI Components & Logic ---

def display_analysis_results(analysis_data):
    """
    Displays the analysis results in a tabbed interface for a single ROI.
    
    Args:
        analysis_data (dict): A dictionary containing all computed results for a specific ROI.
    """
    roi = analysis_data['roi_image']
    enhanced = analysis_data['enhanced_image']
    profile = analysis_data['thermal_profile']
    cool = analysis_data['cooling_rate']
    grad_map = analysis_data['gradient_map']
    GR = analysis_data['GR_ratio']
    microstructure = analysis_data['microstructure']
    haz_width = analysis_data['haz_width']
    stats = analysis_data['meltpool_stats']
    gauss_dict = analysis_data['gaussian_fit_dict']
    gauss_params = analysis_data['gaussian_fit_params']
    residual_map = analysis_data['residual_map']
    crack_risk = analysis_data['crack_risk_index']
    
    st.markdown("### Analysis Results")
    tabs = st.tabs(["ROI Overview", "Image Enhancement", "Thermal Profiles", "Gradient Analysis", 
                    "Melt Pool Geometry", "Scientific Insights", "AI Assistant"])

    with tabs[0]:
        st.subheader("Selected Region of Interest (ROI)")
        st.image(roi, caption="Original ROI", use_column_width=True)
        st.markdown(f"**ROI Dimensions:** {roi.shape[1]}x{roi.shape[0]} pixels")

    with tabs[1]:
        st.subheader("Enhanced Image for Analysis")
        st.image(enhanced, caption="CLAHE Enhanced Image", use_column_width=True)
        st.info(f"CLAHE applied with Clip Limit: {analysis_data['clahe_clip_limit']} and Tile Grid Size: {analysis_data['clahe_tile_grid_size']}")

    with tabs[2]:
        st.subheader("Thermal Profile & Cooling Rate")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(y=profile, mode='lines', name='Thermal Intensity'))
        fig1.update_layout(title="Thermal Profile Across ROI Width",
                           xaxis_title="Pixel Position", yaxis_title="Mean Intensity")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=cool, mode='lines', name='Cooling Rate', line=dict(color='orange')))
        fig2.update_layout(title="Cooling Rate Profile",
                           xaxis_title="Pixel Position", yaxis_title="Intensity Change / Pixel")
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[3]:
        st.subheader("Gradient Map & Microstructure Prediction")
        fig_grad, ax_grad = plt.subplots(figsize=(8, 6))
        im_grad = ax_grad.imshow(grad_map, cmap='plasma', aspect='auto')
        fig_grad.colorbar(im_grad, ax=ax_grad, label='Absolute Gradient Intensity')
        ax_grad.set_title("Absolute Intensity Gradient Map")
        ax_grad.set_xlabel("X (pixels)")
        ax_grad.set_ylabel("Y (pixels)")
        st.pyplot(fig_grad)

        st.markdown(f"### Microstructure Prediction:")
        st.success(f"**Estimated G/R Ratio:** `{GR:.3f}`")
        st.info(f"**Predicted Microstructure:** **`{microstructure}`** growth (based on G/R ratio)")
        st.warning(f"**Estimated HAZ Width:** `{haz_width}` pixels (indicative of heat spread)")

    with tabs[4]:
        st.subheader("Melt Pool Geometry Statistics & Gaussian Fit")
        
        st.markdown("#### Melt Pool Contours Statistics")
        df_stats = pd.DataFrame(stats)
        if not df_stats.empty:
            st.dataframe(df_stats, use_container_width=True)
            download_data_button(df_stats, "meltpool_geometry_stats")
        else:
            st.info("No distinct melt pool contours detected.")
            
        st.markdown("#### 2D Gaussian Fit Parameters")
        df_gauss = pd.DataFrame.from_dict(gauss_dict, orient='index', columns=['Value'])
        st.dataframe(df_gauss, use_container_width=True)
        download_data_button(df_gauss, "gaussian_fit_parameters")

    with tabs[5]:
        st.subheader("Scientific Summary & Shape Deviation")
        summary = generate_scientific_summary(GR, microstructure, cool, gauss_dict, haz_width, crack_risk)
        st.text_area("Comprehensive Scientific Summary", value=summary, height=350)
        
        # Download scientific report button
        st.download_button(
            label="Download Scientific Report (.txt)",
            data=summary.encode('utf-8'),
            file_name=f"ThermalScope_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key=f"download_report_{get_unique_key('report')}"
        )

        st.markdown("#### Shape Deviation (Residual Map)")
        if gauss_params is not None:
            fig_res, ax_res = plt.subplots(figsize=(8, 6))
            im_res = ax_res.imshow(residual_map, cmap='coolwarm', origin='lower',
                                   extent=[0, enhanced.shape[1], 0, enhanced.shape[0]])
            fig_res.colorbar(im_res, ax=ax_res, label='Residual Intensity')
            ax_res.set_title("Residual Map (Actual - Gaussian Fit)")
            ax_res.set_xlabel("X (pixels)")
            ax_res.set_ylabel("Y (pixels)")
            st.pyplot(fig_res)
            st.info("Residual map highlights areas where the actual melt pool deviates from the ideal 2D Gaussian shape.")
        else:
            st.warning("Cannot generate residual map as Gaussian fit failed.")
            
        st.markdown(f"#### Crack Risk Assessment")
        st.error(f"**Estimated Crack Risk Index (CRI):** `{crack_risk:.3f}` (Higher is riskier)")
        st.markdown("*(Note: This is a preliminary, mock index and requires further validation with experimental data.)*")


    with tabs[6]:
        st.subheader("AI Assistant for Melt Pool Insights")
        st.markdown(
            "Leverage the power of AI to get deeper insights and explanations about your melt pool analysis. "
            "The assistant is preloaded with your current melt pool data."
        )

        preload_context = f"You are an AI assistant specialized in metal additive manufacturing and thermal analysis of DED processes. You are analyzing a melt pool region with the following key characteristics: G/R ratio={GR:.3f}, predicted microstructure={microstructure} growth, peak cooling rate={cool.min():.3f}, estimated HAZ width={haz_width} pixels. The melt pool's thermal distribution (Gaussian fit) has Sigma X={gauss_dict.get('Sigma X', 0):.1f} and Sigma Y={gauss_dict.get('Sigma Y', 0):.1f}. The estimated Crack Risk Index is {crack_risk:.3f}. Based on this data, provide insights and answer user questions."
        
        # Ensure chat history exists and system message is the first
        if "chat" not in st.session_state or len(st.session_state.chat) == 0 or st.session_state.chat[0]["role"] != "system":
            st.session_state.chat = [{"role": "system", "content": preload_context}]
        elif st.session_state.chat[0]["content"] != preload_context:
            # Update system message if context changes (e.g., new ROI selected)
            st.session_state.chat[0]["content"] = preload_context
            # Keep only the system message to avoid confusing old context with new analysis
            st.session_state.chat = [st.session_state.chat[0]]

        # Display chat messages
        for message in st.session_state.chat:
            if message["role"] != "system": # Don't display the system message
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        user_query = st.chat_input("Ask about the melt pool or analysis results:")
        if user_query:
            st.session_state.chat.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.spinner("AI Assistant is thinking..."):
                try:
                    # OpenRouter API call
                    # Ensure OPENROUTER_API_KEY is set in environment variables
                    if "OPENROUTER_API_KEY" not in os.environ:
                        st.error("OpenRouter API key not found. Please set the OPENROUTER_API_KEY environment variable.")
                    else:
                        headers = {
                            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                            "HTTP-Referer": "https://thermalscope.streamlit.app", # Replace with your app's URL
                            "X-Title": "ThermalScope Streamlit App",
                        }
                        
                        # Use the entire chat history for context
                        response = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers=headers,
                            json={"model": "openrouter/cypher-alpha:free", "messages": st.session_state.chat},
                        )
                        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                        
                        data = response.json()
                        if 'choices' in data and data['choices']:
                            assistant_reply = data["choices"][0]["message"]
                            st.session_state.chat.append(assistant_reply)
                            with st.chat_message("assistant"):
                                st.markdown(assistant_reply["content"])
                        else:
                            st.error(f"AI Assistant error: No valid response from API. Response: {data}")

                except requests.exceptions.RequestException as e:
                    st.error(f"AI Assistant connection error: {e}. Please check your internet connection or API key.")
                except json.JSONDecodeError:
                    st.error("AI Assistant error: Failed to decode API response. Invalid JSON.")
                except Exception as e:
                    st.error(f"An unexpected error occurred with the AI Assistant: {e}")

    # Data export options for the current ROI
    st.markdown("---")
    st.subheader("Data Export for Current Analysis")
    df_profile = pd.DataFrame({"Pixel": np.arange(len(profile)), "Intensity": profile, "Cooling Rate": cool})
    download_data_button(df_profile, "thermal_profile")
    
    # Collect all dataframes for ZIP export
    all_dfs_for_zip = {
        "thermal_profile": df_profile,
        "meltpool_stats": pd.DataFrame(stats),
        "gaussian_fit": df_gauss
    }
    download_zip_button(all_dfs_for_zip, "current_roi_data")

def process_image_and_roi(img_np, solidification_rate, clahe_clip_limit, clahe_tile_grid_size, roi_coords=None):
    """
    Processes an image (or a specific ROI from it) and performs all scientific analyses.
    
    Args:
        img_np (np.array): The full image as a NumPy array (RGB).
        solidification_rate (float): The user-defined solidification rate.
        clahe_clip_limit (float): CLAHE clip limit.
        clahe_tile_grid_size (int): CLAHE tile grid size.
        roi_coords (tuple, optional): (x, y, width, height) of the ROI. If None, process the whole image.

    Returns:
        dict: A dictionary containing all analysis results.
    """
    if roi_coords:
        x, y, rw, rh = roi_coords
        roi = img_np[y:y+rh, x:x+rw]
        if roi.size == 0: # Handle empty ROI case
            st.error("Selected ROI is empty or invalid. Please draw a valid rectangle.")
            return None
    else:
        roi = img_np # Process the full image if no ROI specified

    # Perform all computations
    enhanced_img = enhance_image(roi, clahe_clip_limit, clahe_tile_grid_size)
    thermal_profile = extract_thermal_profile(enhanced_img)
    cooling_rate = compute_cooling_rate(thermal_profile)
    gradient_map = compute_gradient_map(enhanced_img)
    GR_ratio = compute_GR_ratio(gradient_map, solidification_rate)
    microstructure_prediction = predict_microstructure(GR_ratio)
    haz_width_estimation = estimate_HAZ_width(enhanced_img)
    meltpool_stats = extract_meltpool_geometry_stats(enhanced_img)
    gaussian_fit_dict, gaussian_fit_params = fit_2d_gaussian(enhanced_img)
    # Check if gaussian fit parameters are valid before computing residual map
    if all(p is not None and not np.isnan(p) and not np.isinf(p) for p in gaussian_fit_params):
        residual_map = compute_shape_residual_map(enhanced_img, gaussian_fit_params)
    else:
        residual_map = np.zeros_like(enhanced_img[:,:,0]) # Placeholder if fit failed.
    crack_risk_index = compute_crack_risk_index(GR_ratio, gradient_map, cooling_rate)

    results = {
        'roi_image': roi,
        'enhanced_image': enhanced_img,
        'thermal_profile': thermal_profile,
        'cooling_rate': cooling_rate,
        'gradient_map': gradient_map,
        'GR_ratio': GR_ratio,
        'microstructure': microstructure_prediction,
        'haz_width': haz_width_estimation,
        'meltpool_stats': meltpool_stats,
        'gaussian_fit_dict': gaussian_fit_dict,
        'gaussian_fit_params': gaussian_fit_params,
        'residual_map': residual_map,
        'clahe_clip_limit': clahe_clip_limit,
        'clahe_tile_grid_size': clahe_tile_grid_size,
        'crack_risk_index': crack_risk_index
    }
    return results

def get_roi_selection_html(roi_name, analysis_data):
    """Generates HTML string for ROI selection in comparison mode."""
    gr = analysis_data['GR_ratio']
    micro = analysis_data['microstructure']
    return f"""
    <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
        <h4 style="color: {ACCENT_COLOR};">{roi_name}</h4>
        <p>G/R: <b>{gr:.2f}</b></p>
        <p>Microstructure: <b>{micro}</b></p>
        <img src="data:image/png;base64,{pil_image_to_base64(analysis_data['roi_image'])}" 
             style="max-width: 100px; height: auto; border-radius: 3px;">
    </div>
    """

def pil_image_to_base64(img_array):
    """Converts a NumPy array image to base64 string for embedding in HTML."""
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

import base64 # Import base64 for the pil_image_to_base64 function

# --- Main Streamlit Application Layout ---
def main_app():
    st.markdown("<h1 style='text-align: center; color: #FAFAFA;'>🔬 ThermalScope: Advanced DED Melt Pool Analysis 🔬</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #BBBBBB;'>Intelligent & Publication-Worthy Insights for Metal Additive Manufacturing</p>", unsafe_allow_html=True)
    
    st.sidebar.header("Upload & Global Settings")
    
    # File uploader supports multiple files for frame extrapolation or comparison
    uploaded_files = st.sidebar.file_uploader(
        "Upload Thermal Image(s) / Video Frames",
        type=["png", "jpg", "jpeg", "mp4", "tiff"], # Added mp4 for video support
        accept_multiple_files=True
    )
    
    solidification_rate = st.sidebar.number_input(
        "Solidification Rate (R, units/s)",
        min_value=0.01, max_value=100.0, value=1.0, step=0.1,
        help="Rate at which the material solidifies during cooling."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Image Enhancement Settings (CLAHE)")
    clahe_clip_limit = st.sidebar.slider(
        "CLAHE Clip Limit",
        min_value=1.0, max_value=10.0, value=3.0, step=0.1,
        help="Threshold for contrast limiting. Higher values increase contrast."
    )
    clahe_tile_grid_size = st.sidebar.slider(
        "CLAHE Tile Grid Size",
        min_value=4, max_value=32, value=8, step=1,
        help="Size of the grid for histogram equalization. Smaller grids can enhance local contrast more."
    )

    if uploaded_files:
        # Handle single vs. multiple file uploads
        if len(uploaded_files) > 1:
            st.session_state.compare_mode = st.sidebar.checkbox(
                "Enable Comparison Mode (Select ROIs from different images)", False,
                help="Toggle to analyze and compare multiple images or ROIs side-by-side."
            )
            # Process multiple files/frames
            image_options = {}
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                if file_name not in st.session_state.uploaded_files_data:
                    # Read image as PIL Image, convert to RGB, then to NumPy array
                    img_pil = Image.open(uploaded_file).convert("RGB")
                    img_np = np.array(img_pil)
                    st.session_state.uploaded_files_data[file_name] = img_np
                image_options[file_name] = st.session_state.uploaded_files_data[file_name]
            
            selected_file_name = st.selectbox("Select Image for Current Analysis", list(image_options.keys()))
            img_to_analyze = image_options[selected_file_name]
            st.session_state.current_image_key = selected_file_name # Store key for current image
            
        else: # Single file upload
            st.session_state.compare_mode = False # Disable comparison mode if only one file
            uploaded_file = uploaded_files[0]
            file_name = uploaded_file.name
            if file_name not in st.session_state.uploaded_files_data:
                img_pil = Image.open(uploaded_file).convert("RGB")
                img_np = np.array(img_pil)
                st.session_state.uploaded_files_data[file_name] = img_np
            img_to_analyze = st.session_state.uploaded_files_data[file_name]
            st.session_state.current_image_key = file_name # Store key for current image

        h, w = img_to_analyze.shape[:2]

        st.markdown("---")
        st.subheader("Step 1: Draw Region of Interest (ROI)")
        st.info("Draw a rectangle on the image below to define your Region of Interest for analysis. "
                "The analysis will be performed only on the selected area.")

        # Streamlit Canvas for ROI drawing
        canvas_result = st_canvas(
            stroke_color="#FF0000", # Red stroke for visibility
            fill_color="rgba(255, 0, 0, 0.2)", # Semi-transparent red fill
            background_image=Image.fromarray(img_to_analyze),
            update_streamlit=True,
            height=min(h, 600), # Cap height for responsiveness
            width=min(w, 800),  # Cap width for responsiveness
            drawing_mode="rect",
            key="canvas_main_roi"
        )
        
        selected_roi_coords = None
        if canvas_result.json_data and canvas_result.json_data['objects']:
            obj = canvas_result.json_data['objects'][0]
            # Ensure coordinates are within image bounds and valid
            x = max(0, int(obj['left']))
            y = max(0, int(obj['top']))
            rw = min(int(obj['width']), w - x)
            rh = min(int(obj['height']), h - y)
            
            if rw > 0 and rh > 0:
                selected_roi_coords = (x, y, rw, rh)
            else:
                st.warning("Drawn ROI is too small or invalid. Please draw a larger rectangle.")

        if selected_roi_coords:
            # Generate a unique key for the ROI analysis result
            roi_key = f"{st.session_state.current_image_key}_ROI_{selected_roi_coords[0]}_{selected_roi_coords[1]}"
            
            # Re-run analysis only if ROI or image changed, or if not already analyzed
            if roi_key not in st.session_state.roi_data or \
               st.session_state.roi_data[roi_key].get('clahe_clip_limit') != clahe_clip_limit or \
               st.session_state.roi_data[roi_key].get('clahe_tile_grid_size') != clahe_tile_grid_size or \
               st.session_state.roi_data[roi_key].get('solidification_rate') != solidification_rate:
                
                with st.spinner("Performing advanced melt pool analysis... This may take a moment."):
                    analysis_results = process_image_and_roi(img_to_analyze, solidification_rate, 
                                                            clahe_clip_limit, clahe_tile_grid_size, selected_roi_coords)
                    if analysis_results:
                        # Store additional metadata with the results
                        analysis_results['solidification_rate'] = solidification_rate
                        st.session_state.roi_data[roi_key] = analysis_results
                        st.session_state.current_analysis_key = roi_key # Set the currently displayed analysis
            else:
                st.info("Analysis results for this ROI are already available. Adjust settings or draw new ROI to re-analyze.")
                st.session_state.current_analysis_key = roi_key # Ensure the current key is set

            if st.session_state.get('current_analysis_key') and \
               st.session_state.current_analysis_key in st.session_state.roi_data:
                
                display_analysis_results(st.session_state.roi_data[st.session_state.current_analysis_key])
        else:
            st.info("Draw a rectangle on the image above to begin melt pool analysis.")

    else:
        st.info("👆 Upload thermal image(s) or video frames from your BEAM Magic DED 200 system to get started!")
    
    st.markdown("---")
    st.subheader("Compare Multiple ROIs / Images")

    if st.session_state.compare_mode and st.session_state.roi_data:
        st.markdown("#### Select ROIs to Compare:")
        available_roi_keys = list(st.session_state.roi_data.keys())
        
        # Use a multiselect box to choose ROIs for comparison
        selected_for_comparison = st.multiselect(
            "Choose up to 3 analyses for side-by-side comparison:",
            options=available_roi_keys,
            default=st.session_state.selected_rois_to_compare,
            key="multiselect_compare_rois"
        )
        
        # Update session state for selected ROIs
        st.session_state.selected_rois_to_compare = selected_for_comparison[:3] # Limit to 3 for readability

        if st.session_state.selected_rois_to_compare:
            if len(st.session_state.selected_rois_to_compare) > 3:
                st.warning("Comparison is limited to a maximum of 3 ROIs for better visualization.")

            cols = st.columns(len(st.session_state.selected_rois_to_compare))
            
            for i, roi_key in enumerate(st.session_state.selected_rois_to_compare):
                if roi_key in st.session_state.roi_data:
                    analysis = st.session_state.roi_data[roi_key]
                    with cols[i]:
                        st.markdown(f"**Analysis: {roi_key}**")
                        st.image(analysis['roi_image'], caption="Original ROI", width=150)
                        st.write(f"**G/R Ratio:** {analysis['GR_ratio']:.3f}")
                        st.write(f"**Microstructure:** {analysis['microstructure']}")
                        st.write(f"**Crack Risk Index:** {analysis['crack_risk_index']:.3f}")
                        
                        # Plot mini thermal profile
                        fig_mini_profile = go.Figure()
                        fig_mini_profile.add_trace(go.Scatter(y=analysis['thermal_profile'], mode='lines', name='Thermal Profile'))
                        fig_mini_profile.update_layout(title="Thermal Profile", height=200, margin=dict(l=0, r=0, b=0, t=30),
                                                    xaxis_title="", yaxis_title="")
                        st.plotly_chart(fig_mini_profile, use_container_width=True, config={'displayModeBar': False})
                        
                        # Plot mini gradient map
                        fig_mini_grad, ax_mini_grad = plt.subplots(figsize=(3, 2))
                        ax_mini_grad.imshow(analysis['gradient_map'], cmap='plasma')
                        ax_mini_grad.set_title("Gradient Map", fontsize=10)
                        ax_mini_grad.axis('off')
                        st.pyplot(fig_mini_grad)
                else:
                    with cols[i]:
                        st.error(f"Analysis data for '{roi_key}' not found.")
        else:
            st.info("Select ROIs from the dropdown above to compare their analysis results.")
    elif st.session_state.compare_mode and not st.session_state.roi_data:
        st.info("Upload images and perform analyses first to enable comparison.")
    elif not st.session_state.compare_mode:
        st.info("Toggle 'Enable Comparison Mode' in the sidebar to compare multiple analyses.")


# Run the main application
if __name__ == "__main__":
    main_app()