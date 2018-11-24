import csv
import time
from datetime import datetime
from time import sleep

with open("result.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
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

        sleep(delay * 0.001 * (1 / 50))
        print('{}:\n {}'.format(line[0], line[2]))
