from flask import Flask, render_template
from tensorflow.keras.models import load_model
import pickle
from flask import request, redirect
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html', pred_text = None)

@app.route('/predict',  methods=['POST','GET'] )
def predict():
    if request.method == "POST":
        model = load_model('model.h5')
        pickle_in = open("token_dump","rb")
        tokenizer = pickle.load(pickle_in)
        req = request.form
        seed = req['message'] 
        output_text = ""
        NUM_WORD_GEN = 500
        SEQ = 16
        for i in range(NUM_WORD_GEN):
            encoded_text = tokenizer.texts_to_sequences([seed])[0]
            pad_encoded = pad_sequences([encoded_text], maxlen=SEQ, truncating='pre')
            pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
            pred_word = tokenizer.index_word[pred_word_ind] 
            seed += ' ' + pred_word
            output_text += ' ' + pred_word

        
        print(output_text)
        return render_template('index.html', pred_text = output_text)

    




if __name__ == '__main__':
   app.run(debug = True)

     
