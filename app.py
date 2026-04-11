import json
import os
import streamlit as st
import pandas as pd
import re
import time
import threading
from transformers.pipelines import pipeline

# Try to import sentence_transformers, fallback to simple keyword matching if not available
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# ---- Firebase Setup ----
@st.cache_resource
def get_firestore_db():
    if not firebase_admin._apps:
        creds_str = st.secrets["FIREBASE_CREDENTIALS_JSON"]
        import codecs
        creds_str = codecs.decode(creds_str, 'unicode_escape')
        creds_json = json.loads(creds_str)
        cred = credentials.Certificate(creds_json)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def log_to_firestore(row_dict):
    db = get_firestore_db()
    db.collection("responses").add(row_dict)
# ----------------------------------

STOPWORDS = {'Hassan', 'Asim', 'Ather'}

def looks_like_gibberish(word):
    """Fast heuristic gibberish detection using vowel-consonant patterns."""
    word = word.lower()
    
    if len(word) < 2 or not word.isalpha():
        return True
    
    # Must contain at least one vowel
    if not any(v in word for v in 'aeiou'):
        return True
    
    # Reject 3+ identical repeats
    if re.fullmatch(r".*(.)\1{2,}.*", word):
        return True
    
    # Reject 4+ consonants in a row
    if re.search(r'[^aeiou]{4,}', word):
        return True
    
    # Reject 3+ vowels in a row  
    if re.search(r'[aeiou]{3,}', word):
        return True
    
    return False

def format_cue_word(cue):
    return f"""
    <div style='text-align: center; font-size: 36px; font-weight: bold; color: #010d1a; padding: 20px;'>
        {cue}
    </div>
    """

def format_feedback(msg, color):
    return f"""
    <div style='text-align: center; font-size: 28px; font-weight: bold; color: {color}; padding: 10px; animation: fadeOut 3s ease-in-out forwards;'>
        {msg}
    </div>
    <style>
        @keyframes fadeOut {{
            0% {{ opacity: 1; }}
            70% {{ opacity: 1; }}
            100% {{ opacity: 0.3; }}
        }}
    </style>
    """

def get_safe_progress(current, total):
    if total == 0:
        return 0.0
    return min(max(current / total, 0.0), 1.0)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="tabularisai/robust-sentiment-analysis")

classifier = load_model()

@st.cache_resource
def load_gibberish_detector():
    return pipeline("text-classification", model="madhurjindal/autonlp-Gibberish-Detector-492513457")

gibberish_classifier = load_gibberish_detector()

@st.cache_resource
def load_similarity_model():
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return None

similarity_model = load_similarity_model()

def is_semantically_relevant(response, context, threshold=0.35):
    """Check if response is semantically related to context using embeddings or fallback."""
    if not response or not context:
        return False
    
    # If sentence_transformers available, use AI embeddings
    if SENTENCE_TRANSFORMERS_AVAILABLE and similarity_model:
        emb1 = similarity_model.encode(response, convert_to_tensor=True)
        emb2 = similarity_model.encode(context, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()
        return similarity >= threshold
    
    # Fallback: simple keyword matching
    response_words = set(response.lower().split())
    context_words = set(context.lower().split())
    
    # Check for shared words or if response contains context word
    shared = response_words & context_words
    if shared:
        return True
    
    # Check if any context word is in response
    for word in context_words:
        if len(word) > 3 and word in response.lower():
            return True
    
    return False

os.makedirs("results", exist_ok=True)

@st.cache_data
def load_data_files():
    with open("data/cue_words.txt", "r") as f:
        cue_words = [line.strip() for line in f.readlines()]
    with open("data/sentences.txt", "r") as f:
        sentences = [line.strip() for line in f.readlines()]
    return cue_words, sentences

cue_words, sentences = load_data_files()

if "phase" not in st.session_state:
    st.session_state.user_id = ""
    st.session_state.specific_id = ""
    st.session_state.phase = 0
    st.session_state.step = 0
    st.session_state.score = 0
    st.session_state.used_texts = set()
    st.session_state.responses = []
    st.session_state.start_time = None
    st.session_state.badges = []
    st.session_state.needs_rerun = False

# Check if we need to rerun (set by callback)
if st.session_state.get("needs_rerun"):
    st.session_state.needs_rerun = False
    st.rerun()

st.markdown("""
<style>
body {
    background-color: #f6f9fc;
    color: #222;
}
.stTextInput > div > div > input {
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

if st.session_state.phase == 0:
    st.title("Positive Phrase Intervention")
    st.markdown("""
    Welcome to this two-phase task designed to encourage positive associations and emotional reflection.

    - **Phase 1**: Respond to single cue words with positive and uplifting phrases.
    - **Phase 2**: React to full sentences with encouraging responses.
    - Avoid repeats and generic prepositions.
    """)
    user_input = st.text_input("Enter your Name or Roll Number:")
    specific_id = st.text_input("Enter your Study Participant ID:")
    if st.button("Start Task") and user_input.strip() and specific_id.strip():
        st.session_state.user_id = user_input.strip()
        st.session_state.specific_id = specific_id.strip()
        safe_id = re.sub(r'[^\w\-]', '_', user_input.strip())
        filename = f"results/{safe_id}.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            st.session_state.responses = df.to_dict("records")
            st.session_state.used_texts = set(df["response"].dropna().str.lower().tolist())
            st.session_state.score = df["score"].sum()
            st.session_state.step = sum(1 for r in st.session_state.responses if r["phase"] == 1)
            st.session_state.phase = 2 if st.session_state.step >= len(cue_words) else 1
        else:
            st.session_state.phase = 1
        st.rerun()

if st.session_state.phase == 1:
    st.progress(get_safe_progress(st.session_state.step, len(cue_words)))
    st.markdown(f"**Points**: `{st.session_state.score}` | **Responses**: `{len(st.session_state.used_texts)}`")
    
    # Display badges
    if len(st.session_state.used_texts) >= 10 and "10 Responses" not in st.session_state.badges:
        st.session_state.badges.append("10 Responses")
        st.success("🏅 Badge Earned: 10 Positive Responses!")
    if st.session_state.step >= len(cue_words) and "Phase 1 Master" not in st.session_state.badges:
        st.session_state.badges.append("Phase 1 Master")
        st.success("🏆 Badge Earned: Phase 1 Master!")

    if st.session_state.step < len(cue_words):
        cue = cue_words[st.session_state.step]
        st.markdown(format_cue_word(cue), unsafe_allow_html=True)

        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        feedback = st.empty()

        def handle_input():
            phrase = st.session_state[f"input_{st.session_state.step}"].strip().lower()
            response_time = round(time.time() - st.session_state.start_time, 2)
            result = classifier(phrase)[0]
            label, conf = result['label'], result['score']
            score = calculate_score(label)

            # Fallback for "a nice person"
            if phrase == "a nice person" and label == "NEGATIVE":
                label = "POSITIVE"
                conf = 0.9

            entry = {
                "timestamp": str(datetime.datetime.now()),
                "user": st.session_state.user_id,
                "specific_id": st.session_state.specific_id,
                "phase": 1,
                "cue": cue,
                "sentence": "",
                "response": phrase,
                "sentiment": label,
                "confidence": conf,
                "score": 0,
                "response_time_sec": response_time,
                "accepted": False
            }

            # Check for gibberish using AI model
            gibberish_result = gibberish_classifier(phrase)[0]
            is_gibberish = gibberish_result['label'] == 'gibberish' and gibberish_result['score'] > 0.5

            # Check semantic relevance to cue word
            is_relevant = is_semantically_relevant(phrase, cue, threshold=0.30)

            # Check sentiment (can be neutral or positive, just not negative)
            is_negative = 'negative' in label.lower()

            if phrase in st.session_state.used_texts:
                feedback.markdown(format_feedback("⚠️ Already used! Kindly use a different word", "#e67e22"), unsafe_allow_html=True)
            elif is_gibberish:
                feedback.markdown(format_feedback("❌ That doesn't look like a real word! Try again.", "#c0392b"), unsafe_allow_html=True)
            elif not is_relevant:
                feedback.markdown(format_feedback("❌ Not related to the cue word! Try something relevant.", "#c0392b"), unsafe_allow_html=True)
            elif is_negative:
                feedback.markdown(format_feedback("❌ Negative word detected! Try again.", "#c0392b"), unsafe_allow_html=True)
            else:
                entry["score"] = score
                entry["accepted"] = True
                st.session_state.score += score
                st.session_state.used_texts.add(phrase)
                st.session_state.step += 1
                st.session_state.start_time = None
                feedback.markdown(format_feedback(f"✅ Sentiment: {label} ({conf:.2f}) | Score +{score}", "#27ae60"), unsafe_allow_html=True)

            st.session_state.responses.append(entry)
            
            # Async Firestore write (don't block UI)
            threading.Thread(target=log_to_firestore, args=(entry,), daemon=True).start()
            
            # Flag for rerun (can't call st.rerun() in callback)
            if entry.get("accepted"):
                st.session_state.needs_rerun = True

        st.text_input("Type a related uplifting and positive phrase (up to 3 words):", key=f"input_{st.session_state.step}", on_change=handle_input)

    else:
        st.success("🎉 Congratulations Phase 1 Complete!")
        if st.button("Proceed to Phase 2"):
            st.session_state.step = 0
            st.session_state.phase = 2
            st.rerun()

elif st.session_state.phase == 2:
    st.progress(get_safe_progress(st.session_state.step, len(sentences)))
    st.markdown(f"**Points**: `{st.session_state.score}` | **Responses**: `{len(st.session_state.used_texts)}`")

    if st.session_state.step < len(sentences):
        sentence = sentences[st.session_state.step]
        st.subheader(f"Sentence {st.session_state.step + 1}:")
        st.write(f"**{sentence}**")

        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        feedback = st.empty()

        def handle_input_2():
            phrase = st.session_state[f"input_s2_{st.session_state.step}"].strip().lower()
            response_time = round(time.time() - st.session_state.start_time, 2)
            result = classifier(phrase)[0]
            label, conf = result['label'], result['score']
            score = calculate_score(label)

            # Fallback for "a nice person"
            if phrase == "a nice person" and label == "NEGATIVE":
                label = "POSITIVE"
                conf = 0.9

            entry = {
                "timestamp": str(datetime.datetime.now()),
                "user": st.session_state.user_id,
                "specific_id": st.session_state.specific_id,
                "phase": 2,
                "cue": "",
                "sentence": sentence,
                "response": phrase,
                "sentiment": label,
                "confidence": conf,
                "score": 0,
                "response_time_sec": response_time,
                "accepted": False
            }

            # Check for gibberish using AI model
            gibberish_result = gibberish_classifier(phrase)[0]
            is_gibberish = gibberish_result['label'] == 'gibberish' and gibberish_result['score'] > 0.5

            # Check semantic relevance to sentence context
            is_relevant = is_semantically_relevant(phrase, sentence, threshold=0.25)

            # Check sentiment
            is_negative = 'negative' in label.lower()
            is_positive = 'positive' in label.lower()

            # ACCEPT if: positive sentiment (even if not semantically relevant)
            # REJECT if: negative, gibberish, or neutral AND not relevant
            if phrase in st.session_state.used_texts:
                feedback.markdown(format_feedback("⚠️ Already used! Kindly use something different.", "#e67e22"), unsafe_allow_html=True)
            elif is_gibberish:
                feedback.markdown(format_feedback("❌ That doesn't look like a real word! Try again.", "#c0392b"), unsafe_allow_html=True)
            elif is_negative:
                feedback.markdown(format_feedback("❌ Negative! Try again.", "#c0392b"), unsafe_allow_html=True)
            elif is_positive:
                # Always accept positive sentiment
                entry["score"] = score
                entry["accepted"] = True
                st.session_state.score += score
                st.session_state.used_texts.add(phrase)
                st.session_state.step += 1
                st.session_state.start_time = None
                feedback.markdown(format_feedback(f"✅ Sentiment: {label} ({conf:.2f}) | Score +{score}", "#27ae60"), unsafe_allow_html=True)
            elif not is_relevant:
                feedback.markdown(format_feedback("❌ Not related to the sentence context! Try something relevant.", "#c0392b"), unsafe_allow_html=True)
            else:
                entry["score"] = score
                entry["accepted"] = True
                st.session_state.score += score
                st.session_state.used_texts.add(phrase)
                st.session_state.step += 1
                st.session_state.start_time = None
                feedback.markdown(format_feedback(f"✅ Sentiment: {label} ({conf:.2f}) | Score +{score}", "#27ae60"), unsafe_allow_html=True)

            st.session_state.responses.append(entry)
            
            # Async Firestore write (don't block UI)
            threading.Thread(target=log_to_firestore, args=(entry,), daemon=True).start()
            
            # Flag for rerun (can't call st.rerun() in callback)
            if entry.get("accepted"):
                st.session_state.needs_rerun = True

        st.text_input("Respond with a positive phrase:", key=f"input_s2_{st.session_state.step}", on_change=handle_input_2)

    else:
        if "Intervention Champion" not in st.session_state.badges:
            st.session_state.badges.append("Intervention Champion")
            st.success("🏆 Badge Earned: Intervention Champion!")
        st.session_state.step = 0
        st.session_state.phase = 3
        st.rerun()

elif st.session_state.phase == 3:
    st.balloons()
    st.success("🎉 Congratulations on Completing the Task!")
    st.markdown(f"**Final Score:** `{st.session_state.score}`")
    df = pd.DataFrame(st.session_state.responses)
    st.dataframe(df)

    with st.expander("📊 Click to see Analytics Dashboard"):
        st.subheader("AI Confidence Over Time")
        df["step"] = range(1, len(df) + 1)
        min_step, max_step = st.slider("Select step range:", int(df["step"].min()), int(df["step"].max()), (int(df["step"].min()), int(df["step"].max())))
        filtered_df = df[(df["step"] >= min_step) & (df["step"] <= max_step)]

        # Chart.js for AI Confidence using st.components.v1.html
        chart_data = {
            "type": "line",
            "data": {
                "labels": filtered_df["step"].tolist(),
                "datasets": [{
                    "label": "AI Confidence",
                    "data": filtered_df["confidence"].tolist(),
                    "borderColor": "#27ae60",
                    "backgroundColor": "rgba(39, 174, 96, 0.2)",
                    "fill": True,
                    "tension": 0.4
                }]
            },
            "options": {
                "scales": {
                    "x": {"title": {"display": True, "text": "Step"}},
                    "y": {"title": {"display": True, "text": "Confidence"}, "min": 0, "max": 1}
                },
                "plugins": {
                    "title": {"display": True, "text": "AI Confidence Over Time"}
                }
            }
        }
        chart_html = f"""
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <canvas id="confidenceChart"></canvas>
        <script>
            const ctx = document.getElementById('confidenceChart').getContext('2d');
            new Chart(ctx, {json.dumps(chart_data)});
        </script>
        """
        st.components.v1.html(chart_html, height=400, scrolling=True)

        st.subheader("Score Over Time")
        filtered_df["cumulative"] = filtered_df["score"].cumsum()
        chart_data_score = {
            "type": "line",
            "data": {
                "labels": filtered_df["step"].tolist(),
                "datasets": [{
                    "label": "Cumulative Score",
                    "data": filtered_df["cumulative"].tolist(),
                    "borderColor": "#3498db",
                    "backgroundColor": "rgba(52, 152, 219, 0.2)",
                    "fill": True,
                    "tension": 0.4
                }]
            },
            "options": {
                "scales": {
                    "x": {"title": {"display": True, "text": "Step"}},
                    "y": {"title": {"display": True, "text": "Cumulative Score"}}
                },
                "plugins": {
                    "title": {"display": True, "text": "Score Over Time"}
                }
            }
        }
        chart_html_score = f"""
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <canvas id="scoreChart"></canvas>
        <script>
            const ctx = document.getElementById('scoreChart').getContext('2d');
            new Chart(ctx, {json.dumps(chart_data_score)});
        </script>
        """
        st.components.v1.html(chart_html_score, height=400, scrolling=True)

    st.download_button("Download Results", df.to_csv(index=False).encode(), file_name=f"{st.session_state.user_id}_results.csv")

    if st.button("🔁 Restart"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
