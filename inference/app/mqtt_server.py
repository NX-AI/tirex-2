# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import json
import time

import paho.mqtt.client as mqtt
import requests

from app.config import Settings


class TirexMQTTClient:
    def __init__(self, config: Settings):
        self.config: Settings = config

        self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        if config.mqtt_broker_username is not None:
            self.client.username_pw_set(username=config.mqtt_broker_username, password=config.mqtt_broker_password)

        self.topic_handlers = {
            config.mqtt_topic_forecast: {
                "endpoint": "/univariate/forecast/quantiles",
                "result_topic": config.mqtt_topic_forecast_result,
                "error_topic": config.mqtt_topic_forecast_error,
                "multivariate": False,
            },
            config.mqtt_topic_multivariate_forecast: {
                "endpoint": "/multivariate/forecast/quantiles",
                "result_topic": config.mqtt_topic_multivariate_forecast_result,
                "error_topic": config.mqtt_topic_multivariate_forecast_error,
                "multivariate": True,
            },
        }

        self.client.on_message = self.on_message
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.http_url = f"http://{self.config.http_host}:{self.config.http_port}"

    def on_message(self, client, userdata, msg):
        handler = self.topic_handlers.get(msg.topic)
        if handler is None:
            print(f"Received message on unhandled topic: {msg.topic}")
            return

        msg_id = None
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            msg_id = payload["id"]
            context, prediction_length = payload["context"], payload["prediction_length"]
            quantiles, mean = self.predict(handler, context, prediction_length)

            message = {"id": msg_id, "mean": mean, "quantiles": quantiles}
            self.client.publish(handler["result_topic"], json.dumps(message))
        except Exception as e:
            print(f"Error processing message: {e}")
            message = {"id": msg_id, "error": str(e)}
            self.client.publish(handler["error_topic"], json.dumps(message))

    def predict(self, handler, context, prediction_length):
        response = requests.post(
            f"{self.http_url}{handler['endpoint']}",
            json={"context": context, "prediction_length": prediction_length},
        )
        response.raise_for_status()

        quantiles = response.json()

        mean_quantile_index = 4
        if handler["multivariate"]:
            # quantiles shape: [series][variate][quantile][timestep].
            mean = [[variate[mean_quantile_index] for variate in ts] for ts in quantiles]
        else:
            # quantiles shape: [series][quantile][timestep].
            mean = [ts[mean_quantile_index] for ts in quantiles]

        return quantiles, mean

    def connect(self, keepalive=60):
        try:
            print(f"MQTT is waiting for the HTTP server at {self.http_url} to load the model and go online")
            self.wait_for_api()
            print(f"Connecting to MQTT broker at {self.config.mqtt_broker_host}:{self.config.mqtt_broker_port}")
            self.client.connect(self.config.mqtt_broker_host, self.config.mqtt_broker_port, keepalive)
            self.client.loop_forever()
        finally:
            self.disconnect()

    def disconnect(self):
        self.client.disconnect()
        print("MQTT client disconnected")

    def on_connect(self, client, userdata, connect_flags, reason_code, properties):
        if reason_code == 0:
            print("Connected to MQTT broker")
            for topic in self.topic_handlers:
                client.subscribe(topic)
        else:
            print(f"Failed to connect to MQTT broker with code: {reason_code}")

    def on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        if reason_code != 0:
            print(f"Unexpected disconnection from MQTT broker with code: {reason_code}")

    def wait_for_api(self, timeout=300):
        for _ in range(timeout):
            try:
                response = requests.get(f"{self.http_url}/health")
                if response.status_code == 200:
                    return
            except:
                pass
            time.sleep(1)
        raise TimeoutError(f"MQTT can't connect to {self.http_url} in {timeout} seconds!")
