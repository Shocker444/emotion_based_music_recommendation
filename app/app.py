import streamlit as st
from inference import run_classifier_on_video
from smolagents import DuckDuckGoSearchTool, CodeAgent, HfApiModel
#from inference import run_classifier_on_video
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("🎥 Emotion Detection via Webcam")

st.markdown("Press **Start Webcam** to run face detection and emotion recognition for 100 frames or until you press `q` in the webcam window.")

# Run flag
if "completed" not in st.session_state:
    st.session_state.completed = False
if "dominant_emotion" not in st.session_state:
    st.session_state.dominant_emotion = None

# Button to start
if st.button("▶️ Start Webcam"):
    st.session_state.completed = False
    st.info("⏳ Streaming from your webcam... Please keep your face visible. Press `q` to quit early.")
    dominant = run_classifier_on_video()
    st.session_state.dominant_emotion = dominant
    st.session_state.completed = True

# After webcam run
if st.session_state.completed:
    st.success("✅ Streaming complete!")
    st.markdown(f"### Dominant Detected Emotion: `{st.session_state.dominant_emotion}`")

    emotion_map = {
        'suprise': 'surprise',
        'fear': 'calm',
        'disgust': 'relaxing',
        'happy': 'upbeat',
        'sad': 'comforting',
        'angry': 'chill',
        'neutral': 'easy listening'
    }
    music_platform = st.selectbox("Choose music platform:", ["Spotify", "YouTube Music", "Apple Music"], index=None)

    model = HfApiModel(model_id = "Qwen/Qwen2.5-Coder-32B-Instruct")

    agent = CodeAgent(model = model, tools = [DuckDuckGoSearchTool()])
    if music_platform:
        with st.spinner("Running agent..."):
            st.write(f"Recommeding a list of {emotion_map[st.session_state.dominant_emotion]} from {music_platform.lower()}")
            query = f"Recommend a list of {emotion_map[st.session_state.dominant_emotion]} songs from {music_platform.lower()} with their respective links"
            result = agent.run(query, max_steps=4)
            st.success("Done!")
            st.write(result)
        