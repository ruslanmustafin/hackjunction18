from keras.models import load_model
import numpy as np
from flask import Flask
from flask import render_template
import random

from queue import Queue, Empty
import threading
import pickle
import csv
from datetime import datetime
from time import sleep

from keras.preprocessing import sequence
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from flask import jsonify

class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalar_map = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return clamp(*list(self.scalar_map.to_rgba(val)))

class MovingAverageFilter:
    sequence = []

    def update(self, s, N = 100):
        sequence.extend(s)
        y_filt = []

        if s > N:
            y_filt = np.convolve(sequence, np.ones((N,))/N, mode='valid')

        return y_filt


def load_models(tkn_path, mdl_path):
    model = load_model(mdl_path)
    
    with open(tkn_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    return tokenizer, model


def process_batch(tokenizer, model, batch, timestamps, ma_filter, max_len=60):

    if len(batch) == 0:
        return

    x_seq = tokenizer.texts_to_sequences(batch)
    x = sequence.pad_sequences(x_seq, maxlen=max_len, padding="post", value=0)

    y_score = model.predict(x)

    y_pred = np.ones(y_score.shape[0])

    y_pred[y_score[:,0] > 0.65] = 0
    y_pred[y_score[:,2] > 0.85] = 2

    # y_filt = ma_filter.update(y_pred)
    # y_pred.mean()


    print(f'Processing batch with size {len(batch)}')

    # global_queue.empty()
    # point sentiment, point density
    # global_queue.put((y_pred.mean(), len(batch)))

    global_data.append((y_pred.mean(), len(batch), sum(timestamps)/len(timestamps)))
    
    print(y_pred.mean())

    # return list(y_filt)

def send_messages():

    tokenizer, model = load_models('../sentiment/models/tokenizer.pickle', '../sentiment/models/cnn_sentiment.h5')
    ma_filter = MovingAverageFilter()

    speed = 200

    time_window = 10 * 60 * 1000 #ms
    time_passed = 0
    
    with open("../data/telegram/in.csv", "r") as f:
    
        reader = csv.reader(f, delimiter=",")

        batch = []
        timestamps = []
        

        for i, line in enumerate(reader):
            ddmmyyyy = [int(x) for x in line[1].split(' ')[0].split('.')]
            
            hhmmss = [int(x) for x in line[1].split(' ')[1].split(':')]
            
            dt = datetime(ddmmyyyy[2], ddmmyyyy[1],
                        ddmmyyyy[0], hhmmss[0], hhmmss[1], hhmmss[2])

            milliseconds = int(round(dt.timestamp() * 1000))

            if i == 0:
                prev_time = milliseconds

            delay = milliseconds - prev_time
            prev_time = milliseconds



            sleep(delay * 0.001 * (1 / speed))

            if len(line[2]) > 0:
                batch.append(line[2])
                timestamps.append(milliseconds)
                # print(len(batch), line[2])

            

            time_passed += delay

            msg_queue.put(line[2])

            if time_passed > time_window:
                # process_batch(tokenizer, model, batch, timestamps, ma_filter)
                batch.clear()
                timestamps.clear()

                time_passed = 0


            
            # producer.send('evilpanda', str.encode(
            #     '{}'.format(line[2])))

def clamp(r, g, b, a):
    return int(r*254),int(g*254),int(b*254)#,int(a*254)


msg_queue = Queue()
global_data = []
COL = MplColorHelper('viridis', 0, 2)
app = Flask(__name__)


@app.route("/msg")
def new_messages():
    msgs = []
    while not msg_queue.empty():
        msgs.append(msg_queue.get())

    return jsonify({'data':msgs})
    
@app.route("/batch", methods = ['POST'])
def receive_batch():
    batch = request.json['data']
    print(batch)

@app.route("/")
def chart():
    labels = []#random.sample(range(1, 100), 10)
    values = []#random.sample(range(1, 100), 10)
    messages = []
    colors = []

    # '#%02x%02x%02x' % (0, 128, 64)

    filtered_data = []

    if len(global_data):

        points_to_show = 10
        every_n = 1#int(len(global_data) / points_to_show)

        filtered_data = [x for i,x in enumerate(global_data) if i % every_n == 0]

        for k, i in enumerate(filtered_data):
            labels.append(datetime.utcfromtimestamp(i[2]/1000).strftime('%H:%M'))
            values.append(i[1])
            messages.append('')
            r,g,b,a = cm.jet((i[0]/2)*255)
            colors.append((k/len(global_data), "#{0:02x}{1:02x}{2:02x}ff".format(*COL.get_rgb(i[0]))))
            # colors.append("#000000")

    return render_template('index.html', values=values, labels=labels, colors=colors, messages=messages)

if __name__ == "__main__":
    t = threading.Thread(target=send_messages)
    t.daemon = True
    t.start()

    app.run(host='0.0.0.0', port=5001)                       
    t.join()