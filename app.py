# Python In-built packages
from pathlib import Path
import PIL
import pyttsx3  # Added import for text-to-speech

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# the voice configuration
engine = pyttsx3.init()

# Function to speak the detection results
def speak_detection_results(boxes, engine):

  if len(boxes) > 0:

    # Handle different box formats 
    if boxes.shape[1] == 3:
      labels = boxes[:, -1]
      confidences = boxes[:, -1] 

    elif boxes.shape[1] == 7:
      labels = boxes[:, -2]  
      confidences = boxes[:, -3]

    else:
      print("Unsupported box format")
      return

    for i in range(len(labels)):
      label = labels[i]
      confidence = confidences[i]
      text = f"Label {label} with confidence {confidence:.2f}"
      try:
        engine.say(text)
        engine.runAndWait()
      except:
        pass
    if voice_output_enabled:
        speak_detection_results(boxes, engine) # Pass engine

    # Display performance metrics
    inference_df = res[0].pandas().xyxy[0]  
    inference_time = inference_df['_infer'].iloc[0] / 1000 # in ms
    num_objects = len(boxes)
    st.write(f"Performance Metrics:")
    st.write(f"Image resolution: {image.shape[0]}x{image.shape[1]}") 
    st.write(f"Objects detected: {num_objects}")
    st.write(f"Total processing time: {inference_time:.1f}ms")

    engine.runAndWait()
  
  else:
    print("No objects detected")



# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

# Add a button to control voice output
voice_output_enabled = st.sidebar.checkbox("Enable Voice Output")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                if voice_output_enabled:  # Check if voice output is enabled
                    if len(boxes) > 0:
                        speak_detection_results(boxes, engine) # Speak the results
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
