import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
from gtts import gTTS

# Performans optimizasyonları için TensorFlow ayarı (isteğe bağlı)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Modeli ve tokenizer'ı yükleme
model = load_model('model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Hikaye verilerini yükleme
stories_df = pd.read_csv('etiketli_veri.csv')

# Maksimum sequence uzunluğu (modelin eğitiminde kullanılan uzunluk)
MAX_SEQUENCE_LENGTH = 100

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # JSON verisini alıyoruz
        data = request.get_json(force=True)
        
        if 'input' not in data:
            return jsonify({"error": "No 'input' key found in JSON."}), 400

        input_text = data['input']
        
        if not isinstance(input_text, list):
            return jsonify({"error": "Input must be a list of texts."}), 400

        # Tokenize etme ve padding işlemi
        sequences = tokenizer.texts_to_sequences([' '.join(input_text)])
        input_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        
        # Model tahmini
        predictions = model.predict(input_data)
        predicted_label = int(np.argmax(predictions))

        # O konuya ait bir hikaye seçip yazdıralım
        predicted_story = stories_df[stories_df['topic_label'] == predicted_label].sample(1)['text'].values[0]

        return jsonify({"predicted_topic": predicted_label, "story": predicted_story})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        # JSON verisini alıyoruz
        data = request.get_json(force=True)

        if 'text' not in data:
            return jsonify({"error": "No 'text' key found in JSON."}), 400

        text = data['text']
        
        # gTTS ile seslendirme işlemi
        tts = gTTS(text, lang='en', tld='co.uk')  # İngiliz aksanı için 'co.uk' kullanıyoruz
        tts.save('story.mp3')  # MP3 dosyası olarak kaydediyoruz

        # Dosyayı istemciye gönderiyoruz
        return send_file('story.mp3', as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

































