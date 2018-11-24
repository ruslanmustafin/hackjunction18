from kafka import KafkaProducer
producer = KafkaProducer(
    bootstrap_servers='35.228.26.195:9092', api_version=(0, 10, 1))
for _ in range(100):
    producer.send('evilpanda', b'some message ebytes')
