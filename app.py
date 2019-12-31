from flask import Flask, render_template
from tensorflow.keras.models import load_model
import pickle
from flask import request, redirect
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_mysqldb import MySQL
import yaml

db = yaml.load(open('db.yaml'))


app = Flask(__name__)

app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)



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

        
        
        return render_template('index.html', pred_text = output_text)

    

@app.route('/send_feedback', methods=["POST", "GET"])
def send_feedback():
    if request.method == 'POST':
        req = request.form
        
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users(name, email, comment) VALUES(%s, %s, %s)",(req['Name'], req['Email'], req['Message']))
        mysql.connection.commit()
        cur.close()

    return render_template('index.html')




if __name__ == '__main__':
   app.run(debug = True)

     
