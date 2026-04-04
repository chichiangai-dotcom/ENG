import sys, io, os, asyncio, traceback, json
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from groq import Groq
import edge_tts

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = Flask(__name__, template_folder='../templates')
CORS(app)

RAW_KEY = "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"
client = Groq(api_key=RAW_KEY)

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    try:
        file = request.files['file']
        file_content = file.read()
        
        # 1. 檢查檔案大小，過小視為無聲
        if len(file_content) < 2000: 
            return jsonify({"text": "", "error": "silent"})

        buffer = io.BytesIO(file_content)
        buffer.name = "audio.mp3" 
        
        transcription = client.audio.transcriptions.create(
            file=buffer, model="whisper-large-v3", language="en", response_format="text",
            temperature=0, prompt="Daily English conversation. Return empty if the audio is silent or just noise."
        )
        result = str(transcription).strip()
        
        # 2. 徹底過濾 Whisper 的無聲幻覺 (Thank you syndrome)
        hallucinations = ["thank you", "thanks for watching", "subtitles", "subscribe"]
        if any(h in result.lower() for h in hallucinations) and len(result) < 25:
            result = ""
            
        return jsonify({"text": result})
    except: return jsonify({"error": "STT Failed"}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_msg = data.get("message", "")
        scene = data.get("scenario", "General")
        level = data.get("level", "Intermediate")
        topic = data.get("topic", "General")
        target_word = data.get("target_word", "") # 發音特訓用的目標單字
        
        # 根據等級調整 AI 深度
        level_guide = {
            "Beginner": "Use simple A1/A2 vocabulary and short sentences.",
            "Intermediate": "Use natural B1/B2 conversational English.",
            "Advanced": "Use C1/C2 advanced vocabulary and professional idioms."
        }

        # 針對不同場景設定 AI 大腦
        system_prompt = ""
        
        if scene == "Pronunciation_Eval":
            # 發音特訓模式：比對用戶發音與目標單字，並標示紅字
            system_prompt = (
                f"You are a strict pronunciation judge. The user tried to say the word: '{target_word}'. "
                f"What you heard was: '{user_msg}'. "
                "1. If they are completely correct, praise them. "
                "2. If they are wrong, provide an HTML string highlighting the wrong parts in RED. Example: <span style='color:red; font-weight:bold;'>target_word_with_error_highlighted</span>. "
                "Respond in JSON: {\"reply\": \"Your feedback in English\", \"translation\": \"繁體中文回饋\", \"feedback\": {\"pron_html\": \"Highlighted word or exact word if correct\"}}"
            )
        elif scene == "Assistant":
            # 全能助理模式
            system_prompt = (
                "You are a highly intelligent, omniscient AI Assistant. You can answer ANY question (coding, weather, facts, translation, etc.) freely without acting as a language teacher. "
                "Respond directly to the user's prompt. JSON format: {\"reply\": \"Your full answer\", \"translation\": \"\"}"
            )
        else:
            # 學習與探索模式
            scenarios = {
                "Path": f"an English tutor teaching {topic} at {level} level.",
                "Explore": f"an expert guiding the user through {topic} preparation at {level} level.",
                "General": "a friendly tutor."
            }
            system_prompt = (
                f"You are {scenarios.get(scene, 'a friendly tutor')}. {level_guide.get(level)} "
                "Respond ONLY in valid JSON format. "
                "JSON: {\"reply\": \"English reply\", \"translation\": \"繁體中文翻譯\", \"feedback\": {\"correction\": \"Correction if any, else null\"}}"
            )

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(completion.choices[0].message.content))
    except Exception as e: 
        traceback.print_exc()
        return jsonify({"reply": "System Error. Please try again.", "translation": "系統錯誤，請再試一次。"}), 200

@app.route("/api/tts", methods=["GET"])
def tts():
    text = request.args.get("text", "")
    voice = request.args.get("voice", "en-US-AvaNeural") 
    async def gen():
        comm = edge_tts.Communicate(text, voice)
        data = b""
        async for chunk in comm.stream():
            if chunk["type"] == "audio": data += chunk["data"]
        return data
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    res = loop.run_until_complete(gen())
    return send_file(io.BytesIO(res), mimetype="audio/mpeg")

if __name__ == "__main__": app.run(host="127.0.0.1", port=5000, debug=True)
