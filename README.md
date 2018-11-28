# Evil Panda
VK challenge entry

Track: AI & Big-Data


Sentiment and message density visualization for large chat groups.

To start streaming messages and analyzing the sentiment:

`cd frontend && python service.py --kafka 0`

Set `kafka` to 1 if you want to use kafka queues.

With service running, point you browser to 

`localhost:5001`

to see density plot, and to 

`localhost:5001/msg`

to see messages playback
