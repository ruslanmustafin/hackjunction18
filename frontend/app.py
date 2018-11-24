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
    
    return render_template('index.html', values=values, labels=labels)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)                       
