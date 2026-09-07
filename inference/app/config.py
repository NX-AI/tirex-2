# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "NX-AI/TiRex-2"
    model_device: str = "cpu"

    http_port: int = 8000
    http_host: str = "0.0.0.0"

    mqtt_enabled: int = 0
    mqtt_broker_host: str | None = None
    mqtt_broker_port: int | None = None
    mqtt_broker_username: str | None = None
    mqtt_broker_password: str | None = None
    mqtt_client_id: str = "tirex-worker"
    mqtt_session_expiry: int = 3600

    mqtt_topic_univariate_forecast: str = "tirex/univariate/forecast/request"
    mqtt_topic_multivariate_forecast: str = "tirex/multivariate/forecast/request"
