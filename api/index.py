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
            temperature=0, prompt="Daily English conversation. Return empty if silent or just background noise."
        )
        result = str(transcription).strip()
        
        # 2. 過濾 Whisper 幻覺 (解決沒講話卻出現 Thank you 的問題)
        hallucinations = ["thank you", "thanks for watching", "subtitles", "subscribe", "thanks."]
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
        target_word = data.get("target_word", "")
        
        system_prompt = ""
        
        if scene == "Pronunciation_Eval":
            # 發音特訓：嚴格比對並回傳 is_correct
            system_prompt = (
                f"You are a strict pronunciation judge. The target word is: '{target_word}'. "
                f"The user's speech recognition result is: '{user_msg}'. "
                "1. Determine if the user successfully pronounced the word (allow minor STT misspellings if it sounds identical). Provide boolean 'is_correct'. "
                "2. If incorrect, provide an HTML string highlighting the wrong parts in RED. Example: <span style='color:red; font-weight:bold;'>wrong_part</span>. "
                "Respond in JSON: {\"reply\": \"Encouraging feedback in English\", \"translation\": \"繁體中文回饋\", \"feedback\": {\"pron_html\": \"Highlighted word or correct word\", \"is_correct\": true/false}}"
            )
        elif scene == "Assistant":
            # 自由 AI 專家：不受語言教學限制
            system_prompt = (
                f"You are a world-class expert in: {topic}. You are a helpful AI Assistant. "
                "Answer the user's questions or chat about anything they want based on your expertise. "
                "Do NOT act as an English teacher. Just be a helpful expert. "
                "Respond entirely in Traditional Chinese unless the user asks for English. "
                "JSON format: {\"reply\": \"Your full answer\", \"translation\": \"\"}"
            )
        else:
            # 學習與探索：正常語言教學
            level_guide = {"Beginner": "A1/A2 level.", "Intermediate": "B1/B2 level.", "Advanced": "C1/C2 level."}
            scenarios = {"Path": f"an English tutor for {topic}.", "Explore": f"an expert in {topic} test prep."}
            
            system_prompt = (
                f"You are {scenarios.get(scene, 'a friendly tutor')}. {level_guide.get(level)} "
                "Respond ONLY in JSON. "
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
