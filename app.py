from flask import Flask, render_template
import pickle
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    email = db.Column(db.String(50))
    comment = db.Column(db.String(50))


from flask import request, redirect

from flask_mysqldb import MySQL
import yaml

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'


@app.route('/')
def home():
    return render_template('index.html', pred_text = None)

@app.route('/predict',  methods=['POST','GET'] )
def predict():
    if request.method == "POST":
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import load_model
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

        
        
        return render_template('index.html', pred_text = output_text)

    

@app.route('/send_feedback', methods=["POST", "GET"])
def send_feedback():
    if request.method == 'POST':
        req = request.form
        
        user = User(name=req['Name'], email=req['Email'], comment = req['Message'])
        db.session.add(user)
        db.session.commit()

    return render_template('index.html', pred_text = "Thank You for feedback!:)")


if __name__ == '__main__':
   app.run(debug = True)


     
