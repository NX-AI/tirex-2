# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

import atexit
import multiprocessing as mp
import sys
import time

from app.config import Settings


def run_http():
    import uvicorn

    from app.http_server import app

    settings = Settings()
    uvicorn.run(app, host=settings.http_host, port=settings.http_port)


def run_mqtt():
    from app.mqtt_server import TirexMQTTClient

    settings = Settings()
    client = TirexMQTTClient(settings)
    client.connect()


def main():
    mp.set_start_method("spawn", force=True)
    settings = Settings()

    processes: list[mp.Process] = []

    processes.append(mp.Process(target=run_http, name="HTTP Server"))
    if settings.mqtt_enabled == 1:
        processes.append(mp.Process(target=run_mqtt, name="MQTT Server"))

    def stop_processes():
        for p in processes:
            if p.is_alive():
                p.terminate()

        for p in processes:
            if p.is_alive():
                p.join(timeout=10)

        for p in processes:
            if p.is_alive():
                p.kill()

    atexit.register(stop_processes)

    for p in processes:
        p.start()

    try:
        while True:
            for p in processes:
                if p.exitcode is not None:
                    stop_processes()
                    sys.exit(p.exitcode)
            time.sleep(1)
    except KeyboardInterrupt:
        stop_processes()
        sys.exit(130)


if __name__ == "__main__":
    main()
