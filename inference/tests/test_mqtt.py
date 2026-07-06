# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import json
import time
from queue import Queue
from uuid import uuid4

import paho.mqtt.client as mqtt
import pytest
from paho.mqtt.client import MQTTMessage
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties

from conftest import (
    MEDIAN_QUANTILE_INDEX,
    MULTIVARIATE_SERIES,
    PREDICTION_LENGTH,
    REFERENCE,
    TARGET,
    assert_forecast_close,
    mqtt_host,
    mqtt_port,
)

connect_timeout = 30
test_timeout = 120
Q = MEDIAN_QUANTILE_INDEX


@pytest.fixture(scope="module")
def mqtt_client():
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        protocol=mqtt.MQTTv5,
    )
    try:
        client.connect(mqtt_host, mqtt_port, 60)
        client.loop_start()

        for _ in range(connect_timeout):
            if client.is_connected():
                break
            time.sleep(1)

        if client.is_connected() is False:
            raise ConnectionRefusedError(f"Failed to connect to MQTT broker at {mqtt_host}:{mqtt_port}")

    except Exception as e:
        pytest.fail(f"MQTT Broker connection failed: {e}")

    yield client

    client.loop_stop()
    client.disconnect()


@pytest.fixture
def message_listener(mqtt_client):
    # Fixture that saves received mqtt messages to a queue
    message_queue = Queue()

    def on_message(client, userdata, msg):
        message_queue.put(msg)

    mqtt_client.on_message = on_message

    yield mqtt_client, message_queue

    mqtt_client.on_message = None


def _roundtrip(message_listener, request_topic, context):
    client, message_queue = message_listener

    correlation = str(uuid4())
    reply_topic = f"tirex/test/reply/{correlation}"
    client.subscribe(reply_topic, qos=1)

    props = Properties(PacketTypes.PUBLISH)
    props.ResponseTopic = reply_topic
    props.CorrelationData = correlation.encode()

    request = {"id": correlation, "context": context, "prediction_length": PREDICTION_LENGTH}
    client.publish(request_topic, json.dumps(request), qos=1, properties=props)

    try:
        while True:
            msg: MQTTMessage = message_queue.get(timeout=test_timeout)
            if getattr(msg.properties, "CorrelationData", None) == correlation.encode():
                break
    finally:
        client.unsubscribe(reply_topic)

    assert msg.topic == reply_topic
    payload = json.loads(msg.payload.decode())
    assert "error" not in payload, payload.get("error")
    return payload


def test_mqtt_univariate(message_listener, api_server):
    payload = _roundtrip(
        message_listener,
        "tirex/univariate/forecast/request",
        [TARGET],
    )
    assert_forecast_close(payload["mean"], [REFERENCE.univariate[0, Q]])


def test_mqtt_multivariate(message_listener, api_server):
    payload = _roundtrip(
        message_listener,
        "tirex/multivariate/forecast/request",
        [MULTIVARIATE_SERIES],
    )
    assert_forecast_close(payload["mean"], [REFERENCE.multivariate[:, Q, :]])


def test_mqtt_batch(message_listener, api_server):
    payload = _roundtrip(
        message_listener,
        "tirex/univariate/forecast/request",
        [TARGET, TARGET],
    )
    assert_forecast_close(payload["mean"], [REFERENCE.univariate[0, Q], REFERENCE.univariate[0, Q]])
