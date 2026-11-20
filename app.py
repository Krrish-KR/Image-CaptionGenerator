
import streamlit as st
import os
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

load_dotenv()

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️"
)

st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image and let Azure AI describe it.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image",
             use_container_width=True)

    if st.button("Generate Caption"):

        try:
            endpoint = os.getenv("VISION_ENDPOINT")
            key = os.getenv("VISION_KEY")

            if not endpoint or not key:
                st.error("Please configure Azure credentials in .env")
            else:
                client = ImageAnalysisClient(
                    endpoint=endpoint,
                    credential=AzureKeyCredential(key)
                )

                image_data = uploaded_file.getvalue()

                with st.spinner("Analyzing image using Azure AI..."):
                    result = client.analyze(
                        image_data=image_data,
                        visual_features=[VisualFeatures.CAPTION],
                        gender_neutral_caption=True
                    )

                if result.caption:
                    st.success("Caption Generated Successfully!")
                    st.subheader("AI Generated Caption")
                    st.write(result.caption.text)

                    confidence = result.caption.confidence * 100
                    st.metric("Confidence Score", f"{confidence:.2f}%")
                else:
                    st.warning("No caption was generated.")

        except HttpResponseError as e:
            st.error(f"Azure API Error: {e.message}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
