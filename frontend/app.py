from __future__ import division
from flask import Flask
from flask import Markup
from flask import Flask
from flask import render_template
import random

app = Flask(__name__)
 
@app.route("/")
def chart():
    labels = random.sample(range(1, 100), 10)
    values = random.sample(range(1, 100), 10)

    color = ["#e8ee3a","#403aee","#403aee","#e8ee3a", "#f49080","#e8ee3a","#403aee","#e8ee3a", "#f49080","#403aee"]

    message = ["Foo","","Bar","","","","","Noooo", "",""]

    (lab,val,mes,col) = zip(*sorted(zip(labels, values, message, color), key=lambda x: x[0]))
    
    return render_template('index.html', values=val, labels=lab, messages=mes, colors=col)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)                       
