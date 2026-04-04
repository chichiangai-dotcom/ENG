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
        buffer = io.BytesIO(file.read())
        buffer.name = "audio.wav" 
        
        # --- 核心優化：加強辨識準確度 ---
        transcription = client.audio.transcriptions.create(
            file=buffer, 
            model="whisper-large-v3", 
            language="en", 
            response_format="text",
            temperature=0, # 降低隨機性，讓結果更穩定
            prompt="The user is a student practicing English speaking. Please transcribe the English speech accurately, even if there are minor pronunciation errors."
        )
        return jsonify({"text": str(transcription).strip()})
    except: return jsonify({"error": "STT Failed"}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_msg = data.get("message", "")
        scene = data.get("scenario", "General")
        level = data.get("level", "Intermediate")
        ui_lang = data.get("uiLang", "zh") 
        
        scenarios = {
            "Travel": "a friendly hotel receptionist.",
            "Restaurant": "a waiter.",
            "Interview": "an HR Manager.",
            "Pronunciation": "a strict coach.",
            "General": "a friendly tutor."
        }
        tip_lang = "Traditional Chinese (Taiwan)" if ui_lang == "zh" else "English"

        system_prompt = (
            f"You are a professional English Coach. Scenario: {scenarios.get(scene, 'General')}. "
            f"User Level: {level}. Respond ONLY in JSON format. "
            f"CRITICAL: 'reply' in English, 'translation' in Trad. Chinese, 'tip' in {tip_lang}. "
            "JSON: {\"reply\": \"...\", \"translation\": \"...\", \"feedback\": {\"grammar_score\": 100, \"correction\": \"...\", \"pronunciation\": {\"word\": \"...\", \"ipa\": \"...\", \"tip\": \"...\"}}}"
        )

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(completion.choices[0].message.content))
    except: return jsonify({"reply": "Error", "translation": "錯誤"}), 200

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
