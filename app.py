# app.py
"""
Health Chatbot - Enhanced Streamlit App with Dashboard Navigation
Integrated Phase 1 + Phase 2 + Phase 3 features with improved UI/UX
"""

import streamlit as st
import os
import io
import json
import tempfile
import threading
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Health Chatbot Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #e6f3ff;
        border-left-color: #ff6b6b;
    }
    .bot-message {
        background-color: #f0f7ff;
        border-left-color: #4ecdc4;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ML libs (optional)
try:
    import torch
except Exception:
    torch = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
except Exception:
    pipeline = None

# ONNX runtime for inference if ONNX models available
try:
    import onnxruntime as ort
except Exception:
    ort = None

# STT/TTS optional libs
try:
    import vosk
    VOSK_AVAILABLE = True
except Exception:
    vosk = None
    VOSK_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    gTTS = None
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    pyttsx3 = None
    PYTTSX3_AVAILABLE = False

try:
    import soundfile as sf
    SF_AVAILABLE = True
except Exception:
    sf = None
    SF_AVAILABLE = False

# --------------------------
# Knowledge Base Generator (Phase 1)
# --------------------------
class HealthChatbotDataGenerator:
    def __init__(self):
        self.topics = {
            'hiv_prevention': {'en': 'HIV Transmission Prevention', 'sw': 'Kuzuia Usambazaji wa UKIMWI'},
            'cholera_symptoms': {'en': 'Symptoms of Cholera', 'sw': 'Dalili za Kipindupindu'},
            'handwashing': {'en': 'Importance of Handwashing', 'sw': 'Umuhimu wa Kuosha Mikono'},
            'water_safety': {'en': 'Water Safety and Treatment', 'sw': 'Usalama wa Maji na Matibabu'},
            'malaria_prevention': {'en': 'Malaria Prevention', 'sw': 'Kuzuia Malaria'}
        }
        self.generated_data = {'en': [], 'sw': []}

    def generate_questions_answers(self):
        """Generate comprehensive health Q&A pairs"""
        # HIV Prevention
        hiv_q_en = [
            "How is HIV transmitted?", "What are ways to prevent HIV infection?",
            "Can HIV be spread through mosquito bites?", "How effective are condoms in preventing HIV?"
        ]
        hiv_a_en = [
            "HIV is transmitted through unprotected sex, sharing needles, or mother-to-child during pregnancy, delivery, or breastfeeding.",
            "Prevent HIV by using condoms, regular testing, avoiding needle sharing, and considering PrEP.",
            "No — HIV is not spread by mosquitoes.",
            "When used correctly, condoms are over 99% effective in preventing HIV transmission."
        ]
        hiv_q_sw = [
            "Je, UKIMWI husambazaje?", "Ni njia gani za kuzuia UKIMWI?", 
            "Je, mbu wanaweza kusambaza UKIMWI?", "Kondomu zina ufanisi gani kuzuia UKIMWI?"
        ]
        hiv_a_sw = [
            "UKIMWI husambazwa kwa ngono isiyo salama, kushiriki sindano, au kutoka kwa mama kwa mtoto wakati wa ujauzito, kujifungua au kunyonyesha.",
            "Zuia UKIMWI kwa kondomi, uchunguzi wa mara kwa mara, kuepuka kushiriki sindano, na PrEP.",
            "Hapana — mbu hawawezi kusambaza UKIMWI.",
            "Ikitumika kwa usahihi, kondomi zina ufanisi zaidi ya 99% katika kuzuia usambazaji wa UKIMWI."
        ]

        # Cholera Symptoms
        cholera_q_en = ["What are the symptoms of cholera?", "How quickly do cholera symptoms appear?"]
        cholera_a_en = [
            "Cholera causes severe watery diarrhea, vomiting, leg cramps, and dehydration; stools often look like 'rice water'.",
            "Symptoms can appear within hours up to 5 days, usually 2-3 days."
        ]
        cholera_q_sw = ["Je, ni dalili gani za kipindupindu?", "Dalili zinaonekana lini?"]
        cholera_a_sw = [
            "Dalili ni kuharisha maji makali, kutapika, kukakamaa kwa miguu na kutokwa na maji mwilini.",
            "Dalili zinaweza kuonekana ndani ya masaa machache hadi siku 5."
        ]

        # Handwashing
        hand_q_en = ["Why is handwashing important?", "When should I wash my hands?"]
        hand_a_en = [
            "Handwashing prevents many infections and reduces spread of disease.",
            "Wash hands after using toilet, before eating and after caring for sick people."
        ]
        hand_q_sw = ["Kwa nini kuosha mikono ni muhimu?", "Ni lini ni lazima kuosha mikono?"]
        hand_a_sw = [
            "Kuosha mikono kunazuia maambukizi mengi.",
            "Osha mikono baada ya kutumia choo, kabla ya kula na baada ya kumtunza mgonjwa."
        ]

        # Combine all topics
        def add_pairs(topic_id, q_en, a_en, q_sw, a_sw):
            for i in range(min(len(q_en), len(q_sw))):
                self.generated_data['en'].append({
                    'topic': self.topics[topic_id]['en'],
                    'question': q_en[i].lower(),
                    'answer': a_en[i],
                    'language': 'en',
                    'topic_id': topic_id
                })
                self.generated_data['sw'].append({
                    'topic': self.topics[topic_id]['sw'],
                    'question': q_sw[i].lower(),
                    'answer': a_sw[i],
                    'language': 'sw',
                    'topic_id': topic_id
                })

        add_pairs('hiv_prevention', hiv_q_en, hiv_a_en, hiv_q_sw, hiv_a_sw)
        add_pairs('cholera_symptoms', cholera_q_en, cholera_a_en, cholera_q_sw, cholera_a_sw)
        add_pairs('handwashing', hand_q_en, hand_a_en, hand_q_sw, hand_a_sw)

    def create_knowledge_base(self, path='health_knowledge_base.json'):
        """Create and save knowledge base"""
        knowledge_base = []
        for lang in ['en', 'sw']:
            for item in self.generated_data[lang]:
                knowledge_base.append({
                    'topic': item['topic'],
                    'question': item['question'],
                    'answer': item['answer'],
                    'language': lang,
                    'topic_id': item['topic_id'],
                    'timestamp': datetime.now().isoformat()
                })
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
        return knowledge_base

# --------------------------
# Enhanced Chatbot Class
# --------------------------
class EnhancedHealthChatbot:
    def __init__(self):
        self.knowledge_base = []
        self.current_language = "en"
        self.stt_model = None
        self.qa_pipeline = None
        self.intent_pipeline = None
        self.qa_onnx_sess = None
        self.intent_onnx_sess = None
        self.intent_label_map = None
        self.tts_mode = "gTTS" if GTTS_AVAILABLE else ("pyttsx3" if PYTTSX3_AVAILABLE else None)
        self.conversation_history = []
        self.load_knowledge_base()

    def load_knowledge_base(self, path='health_knowledge_base.json'):
        """Load knowledge base with error handling"""
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                return True
            except Exception as e:
                st.error(f"Error reading knowledge base: {e}")
                self.knowledge_base = []
                return False
        else:
            st.info("Knowledge base not found. Generate one from the Dashboard.")
            self.knowledge_base = []
            return False

    def find_best_match(self, user_question: str):
        """Enhanced matching with semantic similarity"""
        user_words = set(user_question.lower().split())
        best_match = None
        best_score = 0
        
        for entry in self.knowledge_base:
            if entry.get("language", "en") != self.current_language:
                continue
                
            q_words = set(entry.get("question", "").lower().split())
            
            # Calculate word overlap score
            overlap = user_words & q_words
            score = len(overlap)
            
            # Bonus for exact matches
            if user_question.lower() in entry.get("question", "").lower():
                score += 5
                
            if score > best_score:
                best_score = score
                best_match = entry
                
        return best_match if best_score > 1 else None  # Require at least 2 word matches

    # Audio processing methods
    def init_vosk(self, model_path="vosk-model-small-en-us-0.15"):
        """Initialize Vosk STT model"""
        if not VOSK_AVAILABLE:
            return False
        if not os.path.exists(model_path):
            st.warning(f"Vosk model not found at: {model_path}")
            return False
        try:
            self.stt_model = vosk.Model(model_path)
            return True
        except Exception as e:
            st.error(f"Vosk load error: {e}")
            return False

    def speech_to_text_vosk(self, wav_bytes: bytes) -> str:
        """Convert speech to text using Vosk"""
        if self.stt_model is None:
            return ""
        try:
            import wave, json
            wf = wave.open(io.BytesIO(wav_bytes), 'rb')
            rec = vosk.KaldiRecognizer(self.stt_model, wf.getframerate())
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    r = json.loads(rec.Result())
                    results.append(r.get("text", ""))
            final = json.loads(rec.FinalResult()).get("text", "")
            if final:
                results.append(final)
            return " ".join([r for r in results if r]).strip()
        except Exception as e:
            st.error(f"STT error: {e}")
            return ""

    def text_to_speech(self, text: str, lang: str = "en"):
        """Convert text to speech"""
        if not text:
            return None
            
        if GTTS_AVAILABLE:
            try:
                tts = gTTS(text=text, lang=lang if lang in ["en", "sw"] else "en")
                tmp = io.BytesIO()
                tts.write_to_fp(tmp)
                tmp.seek(0)
                return tmp.read()
            except Exception as e:
                st.error(f"gTTS error: {e}")
                
        if PYTTSX3_AVAILABLE:
            try:
                engine = pyttsx3.init()
                tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmpfile.close()
                engine.save_to_file(text, tmpfile.name)
                engine.runAndWait()
                with open(tmpfile.name, "rb") as f:
                    data = f.read()
                os.unlink(tmpfile.name)
                return data
            except Exception as e:
                st.error(f"pyttsx3 error: {e}")
        return None

    # Model loading methods
    def load_transformers_pipelines(self, qa_dir="./fine_tuned_qa_model", intent_dir="./fine_tuned_intent_model"):
        """Load HuggingFace models"""
        if pipeline is None:
            return False
            
        loaded_any = False
        try:
            if os.path.exists(qa_dir):
                self.qa_pipeline = pipeline("question-answering", model=qa_dir, tokenizer=qa_dir)
                loaded_any = True
        except Exception as e:
            st.error(f"QA model error: {e}")
            
        try:
            if os.path.exists(intent_dir):
                self.intent_pipeline = pipeline("text-classification", model=intent_dir, tokenizer=intent_dir)
                # Load intent labels
                map_path = os.path.join(intent_dir, "intent_labels.json")
                if os.path.exists(map_path):
                    with open(map_path, 'r') as f:
                        mapping = json.load(f)
                    self.intent_label_map = {v: k for k, v in mapping.items()}
                loaded_any = True
        except Exception as e:
            st.error(f"Intent model error: {e}")
            
        return loaded_any

    def load_onnx_sessions(self, qa_path="./qa_model.onnx", intent_path="./intent_model.onnx"):
        """Load ONNX models"""
        if ort is None:
            return False
            
        loaded_any = False
        try:
            if os.path.exists(qa_path):
                self.qa_onnx_sess = ort.InferenceSession(qa_path)
                loaded_any = True
        except Exception as e:
            st.error(f"ONNX QA error: {e}")
            
        try:
            if os.path.exists(intent_path):
                self.intent_onnx_sess = ort.InferenceSession(intent_path)
                loaded_any = True
        except Exception as e:
            st.error(f"ONNX Intent error: {e}")
            
        return loaded_any

    def process_question(self, user_question: str) -> Tuple[str, str]:
        """Process user question with multiple fallback strategies"""
        q = user_question.strip()
        if not q:
            return ("Please ask a health-related question.", "Low")
            
        # Add to conversation history
        self.conversation_history.append({
            "role": "user", 
            "content": q, 
            "timestamp": datetime.now().isoformat(),
            "language": self.current_language
        })

        # 1) Try exact KB match first
        kb_match = self.find_best_match(q)
        if kb_match:
            response = kb_match.get("answer", "")
            confidence = "High (KB Match)"
        else:
            # 2) Try intent + QA pipeline
            response, confidence = self._try_advanced_models(q)
            
            # 3) Final fallback
            if not response:
                if self.current_language == "en":
                    response = "I can answer questions about HIV prevention, cholera symptoms, handwashing, and water safety. Please try rephrasing your question."
                else:
                    response = "Naweza kujibu maswali kuhusu kuzuia UKIMWI, dalili za kipindupindu, kuosha mikono, na usalama wa maji. Tafadhali jaribu kuuliza kwa njia tofauti."
                confidence = "Low (Fallback)"

        # Add bot response to history
        self.conversation_history.append({
            "role": "bot", 
            "content": response, 
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "language": self.current_language
        })

        return response, confidence

    def _try_advanced_models(self, question: str) -> Tuple[str, str]:
        """Try advanced model processing"""
        # Intent classification + QA fallback
        if self.intent_pipeline:
            try:
                res = self.intent_pipeline(question)
                if res and isinstance(res, list):
                    label = res[0].get('label', '')
                    score = res[0].get('score', 0)
                    
                    if score > 0.7:  # Confidence threshold
                        # Simple topic-based response
                        topic_responses = {
                            'hiv_prevention': {
                                'en': "HIV can be prevented through safe sex practices, regular testing, and avoiding needle sharing.",
                                'sw': "UKIMWI unaweza kuzuiwa kwa kufanya ngono salama, kupima mara kwa mara, na kuepuka kushiriki sindano."
                            },
                            'cholera_symptoms': {
                                'en': "Cholera symptoms include severe diarrhea, vomiting, and dehydration.",
                                'sw': "Dalili za kipindupindu ni kuharisha sana, kutapika, na upungufu wa maji mwilini."
                            }
                        }
                        
                        for topic_id, responses in topic_responses.items():
                            if topic_id in label.lower():
                                return (responses.get(self.current_language, responses['en']), 
                                       f"Medium (Intent: {score:.2f})")
            except Exception as e:
                st.error(f"Advanced model error: {e}")
                
        return "", ""

    def toggle_language(self) -> str:
        """Toggle between English and Swahili"""
        self.current_language = "sw" if self.current_language == "en" else "en"
        return "Swahili" if self.current_language == "sw" else "English"

    def get_stats(self) -> Dict[str, Any]:
        """Get chatbot statistics"""
        return {
            "total_kb_entries": len(self.knowledge_base),
            "en_entries": len([e for e in self.knowledge_base if e.get('language') == 'en']),
            "sw_entries": len([e for e in self.knowledge_base if e.get('language') == 'sw']),
            "conversation_count": len([m for m in self.conversation_history if m['role'] == 'user']),
            "models_loaded": {
                "qa_pipeline": self.qa_pipeline is not None,
                "intent_pipeline": self.intent_pipeline is not None,
                "stt_model": self.stt_model is not None
            }
        }

# --------------------------
# Streamlit App with Dashboard
# --------------------------
def main():
    # Initialize session state
    if "bot" not in st.session_state:
        st.session_state.bot = EnhancedHealthChatbot()
    if "page" not in st.session_state:
        st.session_state.page = "Chat"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    bot = st.session_state.bot

    # Sidebar - Dashboard Navigation
    st.sidebar.markdown("<h2 style='text-align: center;'>🏥 HealthBot Dashboard</h2>", 
                       unsafe_allow_html=True)
    
    # Navigation
    page = st.sidebar.radio("Navigate to:", 
                           ["Chat", "Knowledge Base", "Models", "Settings", "Analytics"])
    
    # Language toggle
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        st.write(f"**Language:** {'Swahili' if bot.current_language == 'sw' else 'English'}")
    with col2:
        if st.button("Switch"):
            new_lang = bot.toggle_language()
            st.sidebar.success(f"Switched to {new_lang}")

    # System status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    stats = bot.get_stats()
    
    st.sidebar.metric("KB Entries", stats["total_kb_entries"])
    st.sidebar.metric("Conversations", stats["conversation_count"])
    
    # Model status
    st.sidebar.markdown("#### Models")
    for model, loaded in stats["models_loaded"].items():
        status = "✅" if loaded else "❌"
        st.sidebar.write(f"{status} {model.replace('_', ' ').title()}")

    # Main content based on selected page
    if page == "Chat":
        render_chat_page(bot)
    elif page == "Knowledge Base":
        render_knowledge_base_page(bot)
    elif page == "Models":
        render_models_page(bot)
    elif page == "Settings":
        render_settings_page(bot)
    elif page == "Analytics":
        render_analytics_page(bot)

def render_chat_page(bot):
    """Render the main chat interface"""
    st.markdown("<h1 class='main-header'>💬 Health Chatbot</h1>", 
               unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for msg in st.session_state.chat_history[-10:]:  # Show last 10 messages
            if msg[0] == "You":
                st.markdown(f"""
                <div class='chat-message user-message'>
                    <strong>👤 You:</strong> {msg[1]}<br>
                    <small>{msg[2]}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='chat-message bot-message'>
                    <strong>🤖 Bot:</strong> {msg[1]}<br>
                    <small>{msg[2]} | Confidence: {msg[3] if len(msg) > 3 else 'N/A'}</small>
                </div>
                """, unsafe_allow_html=True)

    # Input area
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_input("Type your health question:", 
                                 placeholder="Ask about HIV, cholera, hygiene...")
    
    with col2:
        send_btn = st.button("Send", use_container_width=True)
    
    # Audio upload
    with st.expander("🎤 Audio Input (Optional)"):
        audio_file = st.file_uploader("Upload audio question", 
                                    type=["wav", "mp3", "m4a", "ogg"])
        
        if audio_file is not None:
            st.audio(audio_file)
            if st.button("Transcribe & Send"):
                with st.spinner("Transcribing..."):
                    audio_bytes = audio_file.read()
                    if bot.stt_model:
                        transcription = bot.speech_to_text_vosk(audio_bytes)
                        if transcription:
                            user_input = transcription
                            send_btn = True
                        else:
                            st.error("Could not transcribe audio")
                    else:
                        st.warning("STT model not loaded")

    # Process message
    if send_btn and user_input:
        process_user_message(bot, user_input)

def render_knowledge_base_page(bot):
    """Render knowledge base management page"""
    st.markdown("<h1>📚 Knowledge Base</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Current Knowledge Base")
        if bot.knowledge_base:
            df = pd.DataFrame(bot.knowledge_base)
            st.dataframe(df[['topic', 'question', 'language']], use_container_width=True)
        else:
            st.info("No knowledge base loaded")
    
    with col2:
        st.subheader("Manage KB")
        
        # Generate new KB
        if st.button("Generate Basic KB"):
            with st.spinner("Generating knowledge base..."):
                gen = HealthChatbotDataGenerator()
                gen.generate_questions_answers()
                kb = gen.create_knowledge_base()
                bot.load_knowledge_base()
                st.success(f"Generated {len(kb)} entries!")
        
        # Upload custom KB
        uploaded_file = st.file_uploader("Upload KB JSON", type=['json'])
        if uploaded_file:
            try:
                kb_data = json.load(uploaded_file)
                with open('health_knowledge_base.json', 'w') as f:
                    json.dump(kb_data, f)
                bot.load_knowledge_base()
                st.success("KB uploaded successfully!")
            except Exception as e:
                st.error(f"Error uploading KB: {e}")
        
        # Statistics
        if bot.knowledge_base:
            st.metric("Total Entries", len(bot.knowledge_base))
            en_count = len([e for e in bot.knowledge_base if e['language'] == 'en'])
            sw_count = len([e for e in bot.knowledge_base if e['language'] == 'sw'])
            st.metric("English Entries", en_count)
            st.metric("Swahili Entries", sw_count)

def render_models_page(bot):
    """Render model management page"""
    st.markdown("<h1>🤖 AI Models</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("HuggingFace Models")
        
        if st.button("Load HF Models"):
            with st.spinner("Loading models..."):
                if bot.load_transformers_pipelines():
                    st.success("HF models loaded!")
                else:
                    st.error("Failed to load HF models")
        
        # Display HF model status
        st.write("**QA Pipeline:**", "✅ Loaded" if bot.qa_pipeline else "❌ Not loaded")
        st.write("**Intent Pipeline:**", "✅ Loaded" if bot.intent_pipeline else "❌ Not loaded")
    
    with col2:
        st.subheader("ONNX Models")
        
        if st.button("Load ONNX Models"):
            with st.spinner("Loading ONNX models..."):
                if bot.load_onnx_sessions():
                    st.success("ONNX models loaded!")
                else:
                    st.error("Failed to load ONNX models")
        
        # Display ONNX status
        st.write("**QA ONNX:**", "✅ Loaded" if bot.qa_onnx_sess else "❌ Not loaded")
        st.write("**Intent ONNX:**", "✅ Loaded" if bot.intent_onnx_sess else "❌ Not loaded")
    
    # STT Configuration
    st.markdown("---")
    st.subheader("Speech-to-Text")
    
    vosk_path = st.text_input("Vosk Model Path", value="vosk-model-small-en-us-0.15")
    if st.button("Initialize Vosk"):
        if bot.init_vosk(vosk_path):
            st.success("Vosk model initialized!")
        else:
            st.error("Failed to initialize Vosk")

def render_settings_page(bot):
    """Render settings page"""
    st.markdown("<h1>⚙️ Settings</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("TTS Settings")
        tts_option = st.selectbox("TTS Engine", 
                                ["gTTS (Online)", "pyttsx3 (Offline)"],
                                index=0 if bot.tts_mode == "gTTS" else 1)
        
        st.subheader("Audio Settings")
        st.slider("Audio recording duration (seconds)", 3, 10, 5)
    
    with col2:
        st.subheader("Display Settings")
        st.checkbox("Show confidence scores", value=True)
        st.checkbox("Play TTS responses automatically", value=True)
        st.checkbox("Save conversation history", value=True)
    
    # System info
    st.markdown("---")
    st.subheader("System Information")
    
    sys_info = {
        "Python Version": sys.version.split()[0],
        "Streamlit Version": st.__version__,
        "GPU Available": torch.cuda.is_available() if torch else False,
        "Vosk Available": VOSK_AVAILABLE,
        "gTTS Available": GTTS_AVAILABLE,
        "ONNX Runtime Available": ort is not None
    }
    
    for key, value in sys_info.items():
        st.write(f"**{key}:** {value}")

def render_analytics_page(bot):
    """Render analytics page"""
    st.markdown("<h1>📊 Analytics</h1>", unsafe_allow_html=True)
    
    stats = bot.get_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Questions", stats["conversation_count"])
        st.metric("KB Entries", stats["total_kb_entries"])
    
    with col2:
        st.metric("English Entries", stats["en_entries"])
        st.metric("Swahili Entries", stats["sw_entries"])
    
    with col3:
        st.metric("Active Models", sum(stats["models_loaded"].values()))
        st.metric("Current Language", bot.current_language.upper())
    
    # Conversation history
    st.subheader("Recent Conversations")
    if bot.conversation_history:
        recent_chats = bot.conversation_history[-5:]  # Last 5 exchanges
        for chat in recent_chats:
            role_icon = "👤" if chat["role"] == "user" else "🤖"
            st.write(f"{role_icon} **{chat['role'].title()}** ({chat['timestamp'][11:19]}):")
            st.write(f"_{chat['content']}_")
            st.write("---")
    else:
        st.info("No conversation history yet")

def process_user_message(bot, message):
    """Process user message and update chat"""
    # Add user message to history
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append(("You", message, timestamp))
    
    # Get bot response
    with st.spinner("Thinking..."):
        response, confidence = bot.process_question(message)
        
        # Add bot response to history
        st.session_state.chat_history.append(("Bot", response, timestamp, confidence))
        
        # Text-to-speech
        if GTTS_AVAILABLE or PYTTSX3_AVAILABLE:
            tts_bytes = bot.text_to_speech(response, lang=bot.current_language)
            if tts_bytes:
                st.audio(tts_bytes)
    
    # Refresh the display
    st.rerun()

if __name__ == "__main__":
    main()
