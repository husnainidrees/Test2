from flask import Flask, request, jsonify
import speech_recognition as sr

app = Flask(__name__)

@app.route('/audio-to-text', methods=['POST'])
def audio_to_text():
    if 'Audio' not in request.files:
        return jsonify({'error': 'No audio file found'}), 400

    audio_file = request.files['Audio']
    
    recognizer = sr.Recognizer()
    
    try:
        # Audio ko SpeechRecognition ke format mein convert karo
        audio = sr.AudioFile(audio_file)
        with audio as source:
            audio_data = recognizer.record(source)
        
        # Speech ko text mein convert karo
        text = recognizer.recognize_google(audio_data)
        return jsonify({'text': text}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)