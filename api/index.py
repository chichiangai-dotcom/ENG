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
        context_word = request.form.get('context_word', '') # 只接收純英文單字
        scenario = request.form.get('scenario', '')
        
        file_content = file.read()
        if len(file_content) < 2000: 
            return jsonify({"text": "", "error": "silent"})

        buffer = io.BytesIO(file_content)
        buffer.name = "audio.mp3" 
        
        # 🔥 Whisper 防幻覺提示詞設定
        prompt_text = "Hello, this is a daily English conversation."
        if scenario == 'Pronunciation_Eval' and context_word:
            prompt_text = context_word # 發音特訓時，只給目標英文單字當提示

        transcription = client.audio.transcriptions.create(
            file=buffer, model="whisper-large-v3", language="en", response_format="text",
            temperature=0.0, prompt=prompt_text
        )
        result = str(transcription).strip()
        
        # 🔥 終極幻覺過濾器 (去除噪音腦補)
        lower_result = result.lower().replace('.', '').replace('!', '').replace('?', '').strip()
        lower_prompt = prompt_text.lower().replace('.', '').strip()
        
        hallucinations = ["thank you", "thanks for watching", "subtitles", "subscribe", "thanks", "hello", "旅遊通關", "科技電腦", "日常社交"]
        
        # 如果辨識結果是空的、是常見幻覺、或者在沒說話時直接吐出 prompt_text，就當作沒聲音
        if not lower_result or \
           any(h in lower_result for h in hallucinations) or \
           (scenario == 'Pronunciation_Eval' and lower_result == lower_prompt and len(file_content) < 10000): 
            # 檔案太小卻精準辨識出單字，通常是幻覺
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
        lesson_num = data.get("lesson_num", 1)
        
        system_prompt = ""
        
        if scene == "Pronunciation_Eval":
            system_prompt = (
                f"You are a strict pronunciation judge. Target word: '{target_word}'. "
                f"User's speech: '{user_msg}'. "
                "1. Determine if the user successfully pronounced it. Provide boolean 'is_correct'. "
                "2. If incorrect, provide HTML highlighting the wrong parts in RED. Example: <span style='color:red; font-weight:bold;'>wrong_part</span>. "
                "JSON: {\"reply\": \"Encouraging feedback in English\", \"translation\": \"繁體中文回饋\", \"feedback\": {\"pron_html\": \"Highlighted word\", \"is_correct\": true/false}}"
            )
        elif scene == "Assistant":
            system_prompt = (
                f"You are a world-class expert in: {topic}. Answer the user's questions freely. "
                "Respond entirely in Traditional Chinese unless asked for English. "
                "JSON format: {\"reply\": \"Your answer\", \"translation\": \"\"}"
            )
        else:
            level_guide = {"Beginner": "A1/A2.", "Intermediate": "B1/B2.", "Advanced": "C1/C2."}
            scenarios = {"Path": f"an English tutor teaching Lesson {lesson_num} of 10 about {topic}.", "Explore": f"an expert test examiner in {topic}."}
            
            system_prompt = (
                f"You are {scenarios.get(scene, 'a friendly tutor')}. Level: {level_guide.get(level)} "
                f"CRITICAL INSTRUCTIONS: "
                f"1. STRICTLY keep the conversation on the topic of '{topic}'. "
                f"2. ERROR CORRECTION: If the user's input is completely off-topic or seems like a speech recognition error, politely explain what they might have gotten wrong, ask them to repeat it clearly, and guide them back to '{topic}'. "
                "3. You MUST actively lead the conversation. End your reply with a direct question to keep them talking. "
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
