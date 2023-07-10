from flask import Flask,render_template,redirect,url_for,request
import numpy as np
import pickle
model=pickle.load(open("finalized_model.pkl", 'rb'))  #before model load open model file in readbinary mode
vector=pickle.load(open("vectorizer.pkl",'rb'))
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=='POST':
       newstxt=request.form['news']
       print(newstxt)
       #transfrom the news
       tf_news=vector.transform([newstxt])
       pred=model.predict(tf_news)[0]
       print(pred)
       return render_template('prediction.html',predicted_text="News is ->{}".format(pred))
    else:
        return redirect('/prediction')
       
    
    
if __name__=='__main__':
    app.run(debug=True)