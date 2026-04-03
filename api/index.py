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
        buffer.name = "speech.wav" 
        transcription = client.audio.transcriptions.create(
            file=buffer, model="whisper-large-v3", language="en", response_format="text"
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
        
        # 擴充的實用情境
        scenarios = {
            "Travel": "a friendly hotel receptionist or tourist guide helping the user on their vacation.",
            "Restaurant": "a waiter at a popular local restaurant taking the user's order.",
            "Interview": "an HR Manager interviewing the user for a professional job.",
            "Pronunciation": "a strict pronunciation coach. Give the user ONE short English sentence to read aloud, and evaluate their pronunciation.",
            "General": "a friendly English tutor for casual free conversation."
        }

        # 根據介面語系決定「解說提示」的語言，但「主文」強制英文，「翻譯」強制中文
        tip_lang = "Traditional Chinese (Taiwan)" if ui_lang == "zh" else "English"

        system_prompt = (
            f"You are a top-tier English Coach. Current Scenario: {scenarios.get(scene, 'General')}. "
            f"User's Level: {level}. "
            "Respond ONLY in valid JSON format. "
            "CRITICAL RULES: "
            "1. 'reply' MUST ALWAYS be in English. "
            "2. 'translation' MUST ALWAYS be the Traditional Chinese translation of the reply. "
            f"3. 'tip' MUST be written in {tip_lang}. "
            "JSON structure: "
            "{"
            "  \"reply\": \"English spoken response\","
            "  \"translation\": \"繁體中文翻譯\","
            "  \"feedback\": {"
            "    \"grammar_score\": 100,"
            "    \"correction\": \"Corrected sentence (or null)\","
            "    \"pronunciation\": {"
            "      \"word\": \"Target word\","
            "      \"ipa\": \"IPA symbol\","
            "      \"tip\": \"Pronunciation explanation\""
            "    }"
            "  }"
            "}"
        )

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
            response_format={"type": "json_object"}
        )
        return jsonify(json.loads(completion.choices[0].message.content))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"reply": "I had a system hiccup. Can you repeat?", "translation": "系統出錯，能重說一次嗎？"}), 200

@app.route("/api/tts", methods=["GET"])
def tts():
    text = request.args.get("text", "")
    voice = request.args.get("voice", "en-US-AvaNeural") 
    if not text: return "No text", 400
    
    async def gen():
        comm = edge_tts.Communicate(text, voice)
        data = b""
        async for chunk in comm.stream():
            if chunk["type"] == "audio": data += chunk["data"]
        return data
        
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        res = loop.run_until_complete(gen())
        loop.close()
        return send_file(io.BytesIO(res), mimetype="audio/mpeg")
    except: return "TTS Error", 500

if __name__ == "__main__": app.run(host="127.0.0.1", port=5000, debug=True)