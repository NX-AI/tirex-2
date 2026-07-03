# TiRex Inference Server

This docker container runs the TiRex model and provides the following APIs to interact with the model:
- **HTTP API**
- **MQTT**
- **MCP**

## Using the docker container

### HTTP API

Run the CPU container:
```
docker run -p 8000:8000 -it ghcr.io/nx-ai/tirex2-cpu
```

Run the GPU container:
```
docker run --gpus all -p 8000:8000 -it ghcr.io/nx-ai/tirex2-gpu
```

Both the CPU and GPU containers run a warmup forecast on startup so the model is compiled before the first request. torch.compile generates kernels for parts of the model (C++ on CPU, Triton on GPU), and on GPU it also JIT-compiles FlashRNN's native CUDA kernels. Warmup can take up to ~10-20 seconds.

When the container is running and has the model loaded, it exposes the HTTP API at [http://localhost:8000/](http://localhost:8000/). Swagger documentation of the API is provided at [http://localhost:8000/docs](http://localhost:8000/docs).

**Bash:**
```bash

# Univariate series
curl -s -X POST "http://localhost:8000/univariate/forecast/mean" \
  -H 'Content-Type: application/json' \
  -d '{
        "context": [[1, 2, 3, 4, 5, 6, 7, 8]],
        "prediction_length": 5
      }'

# Univariate batch (batch size 2): two independent series forecast in a single request
curl -s -X POST "http://localhost:8000/univariate/forecast/mean" \
  -H 'Content-Type: application/json' \
  -d '{
        "context": [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
        "prediction_length": 5
      }'

# Multivariate (multi-target) series
curl -s -X POST "http://localhost:8000/multivariate/forecast/mean" \
  -H 'Content-Type: application/json' \
  -d '{
        "context": [{"target": [[1, 2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 40, 50, 60, 70, 80]]}],
        "prediction_length": 5
      }'

# Multivariate batch (batch size 2): two independent multi-target series in a single request
curl -s -X POST "http://localhost:8000/multivariate/forecast/mean" \
  -H 'Content-Type: application/json' \
  -d '{
        "context": [
          {"target": [[1, 2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 40, 50, 60, 70, 80]]},
          {"target": [[8, 7, 6, 5, 4, 3, 2, 1], [80, 70, 60, 50, 40, 30, 20, 10]]}
        ],
        "prediction_length": 5
      }'

# Multivariate with future covariates
curl -s -X POST "http://localhost:8000/multivariate/forecast/mean" \
  -H 'Content-Type: application/json' \
  -d '{
        "context": [{
          "target": [[1, 2, 3, 4, 5, 6, 7, 8]],
          "future_covariates": [[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]]
        }],
        "prediction_length": 5
      }'

# Multivariate with future covariates, batch (batch size 2): two independent series in a single request
curl -s -X POST "http://localhost:8000/multivariate/forecast/mean" \
  -H 'Content-Type: application/json' \
  -d '{
        "context": [
          {
            "target": [[1, 2, 3, 4, 5, 6, 7, 8]],
            "future_covariates": [[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]]
          },
          {
            "target": [[8, 7, 6, 5, 4, 3, 2, 1]],
            "future_covariates": [[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]]
          }
        ],
        "prediction_length": 5
      }'
```

**Python:**
```python
import requests

# Univariate series
resp = requests.post(
    f"http://localhost:8000/univariate/forecast/mean",
    json={"context": [[1, 2, 3, 4, 5, 6, 7, 8]], "prediction_length": 5},
)
print(resp.json())

# Univariate batch (batch size 2): two independent series forecast in a single request
resp = requests.post(
    f"http://localhost:8000/univariate/forecast/mean",
    json={"context": [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]], "prediction_length": 5},
)
print(resp.json())

# Multivariate (multi-target) series
resp = requests.post(
    f"http://localhost:8000/multivariate/forecast/mean",
    json={
        "context": [{"target": [[1, 2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 40, 50, 60, 70, 80]]}],
        "prediction_length": 5,
    },
)
print(resp.json())

# Multivariate batch (batch size 2): two independent multi-target series in a single request
resp = requests.post(
    f"http://localhost:8000/multivariate/forecast/mean",
    json={
        "context": [
            {"target": [[1, 2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 40, 50, 60, 70, 80]]},
            {"target": [[8, 7, 6, 5, 4, 3, 2, 1], [80, 70, 60, 50, 40, 30, 20, 10]]},
        ],
        "prediction_length": 5,
    },
)
print(resp.json())

# Multivariate with future covariates
resp = requests.post(
    f"http://localhost:8000/multivariate/forecast/mean",
    json={
        "context": [{
            "target": [[1, 2, 3, 4, 5, 6, 7, 8]],
            "future_covariates": [[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]],
        }],
        "prediction_length": 5,
    },
)
print(resp.json())

# Multivariate with future covariates, batch (batch size 2): two independent series in a single request
resp = requests.post(
    f"http://localhost:8000/multivariate/forecast/mean",
    json={
        "context": [
            {
                "target": [[1, 2, 3, 4, 5, 6, 7, 8]],
                "future_covariates": [[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]],
            },
            {
                "target": [[8, 7, 6, 5, 4, 3, 2, 1]],
                "future_covariates": [[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]],
            },
        ],
        "prediction_length": 5,
    },
)
print(resp.json())
```

Every request is batched, so provide a list of timeseries as context, even when you only forecast a single timeseries. Bigger batch sizes are more efficient for the hardware, but too big batch sizes can lead to out of memory errors. There isn't any internal batching done, so the consumer of the API is responsible to call with an appropriate batch size for the hardware.

The HTTP API also provides `/univariate/forecast/quantiles` and `/multivariate/forecast/quantiles`, where the 10, 20, 30, 50 (mean), 60, 70, 80 and 90% quantiles are returned, using the same arguments as the `/univariate/forecast/mean` and `/multivariate/forecast/mean` endpoints respectively.

### MQTT API
The MQTT integration uses **MQTT v5** with a request/reply pattern. TiRex subscribes to the forecast **request** topics and publishes each result back to the **response topic the requester specifies on the request** (the MQTT v5 `Response Topic` property). Every client therefore receives only its own results — there is no shared result topic.

To use the MQTT API you need a **v5-capable** MQTT broker running. For some quick testing, a public test MQTT broker like [broker.emqx.io](https://broker.emqx.io) works (Do not send sensitive data to a public broker!). For testing we also use the [MQTTX CLI](https://mqttx.app/cli).

Start the container with MQTT:
```
docker run -p 8000:8000 -it -e MQTT_ENABLED=1 -e MQTT_BROKER_HOST=broker.emqx.io -e MQTT_BROKER_PORT=1883 ghcr.io/nx-ai/tirex2-cpu
```

Each request must be sent over MQTT v5 and set a **Response Topic** telling TiRex where to publish the result. Optionally set **Correlation Data** to match the reply back to the request. Requests without a Response Topic are rejected.

Subscribe to your own reply topic (choose any topic unique to your client), over MQTT v5:
```
mqttx sub -V 5 -t 'tirex/my-client/result' -h 'broker.emqx.io' -p 1883
```

Send a forecast request, pointing its Response Topic at that reply topic:
```
mqttx pub -V 5 \
  -t 'tirex/univariate/forecast/request' \
  --response-topic 'tirex/my-client/result' \
  --correlation-data '1234' \
  -h 'broker.emqx.io' -p 1883 \
  -m '{"id": "1234", "context": [[0, 1, 2, 3]], "prediction_length": 4}'
```

The result is published to your Response Topic, with the Correlation Data echoed back. Successful results contain `mean` and `quantiles`; if an error happens during processing, the message published to the same Response Topic contains an `error` field instead.

### MCP

Start the docker container as explained in the HTTP API section.

To connect the MCP to a tool like Claude Desktop, follow their [guide](https://modelcontextprotocol.io/docs/develop/connect-local-servers). The following `tirex` line has to be added to the `claude_desktop_config.json` under `mcpServers`:
```json
{
  "mcpServers": {
    "tirex": { "command": "npx", "args": ["-y", "mcp-remote", "http://127.0.0.1:8000/mcp"] }
  }
}
```

## Configuration Options

You can set these env variables when running the container using the -e env flag, like `docker run -e MODEL_DEVICE=cuda ghcr.io/nx-ai/tirex2-cpu`

| Environment Variable                        | Default Value                         | Description                                                     |
| :------------------------------------------ | :------------------------------------ | :-------------------------------------------------------------- |
| **MODEL_PATH**                              | `NX-AI/TiRex-2`                       | The Huggingface model id.                                       |
| **MODEL_DEVICE**                            | `cpu`                                 | Device to run the model on (`cpu` or `cuda`).                   |
| **HTTP_HOST**                               | `0.0.0.0`                             | Host the HTTP server binds to.                                  |
| **HTTP_PORT**                               | `8000`                                | Port the HTTP server binds to.                                  |
| **MQTT_ENABLED**                            | `0`                                   | Enable MQTT client functionality (1=True, 0=False).             |
| **MQTT_BROKER_HOST**                        | `None`                                | Hostname or IP address of the MQTT broker.                      |
| **MQTT_BROKER_PORT**                        | `None`                                | Port of the MQTT broker.                                        |
| **MQTT_BROKER_USERNAME**                    | `None`                                | Username for authenticating with the MQTT broker (if required). |
| **MQTT_BROKER_PASSWORD**                    | `None`                                | Password for authenticating with the MQTT broker (if required). |
| **MQTT_CLIENT_ID**                          | `tirex-worker`                        | Stable, unique client id so the broker can resume the session on reconnect. |
| **MQTT_SESSION_EXPIRY**                     | `3600`                                | Seconds the broker retains the session (and queued requests) while disconnected. |
| **MQTT_TOPIC_UNIVARIATE_FORECAST**          | `tirex/univariate/forecast/request`   | Topic to subscribe to for univariate forecast requests.         |
| **MQTT_TOPIC_MULTIVARIATE_FORECAST**        | `tirex/multivariate/forecast/request` | Topic to subscribe to for multivariate forecast requests.       |


## Build and run the docker container

`NX-AI/TiRex-2` is a gated Hugging Face repo, so provide a read access token via `HF_TOKEN` to download the weights — either from your shell (`export HF_TOKEN=hf_xxxxxxxx`) or via an `--env-file`.

### CPU Container

Build the CPU image:
```
docker build -f Dockerfile.cpu -t tirex2-inference-cpu .
```

Run the CPU container:
```
docker run --rm -p 8000:8000 \
  -e HF_TOKEN \
  -v tirex-hf-cache:/home/appuser/.cache/huggingface \
  tirex2-inference-cpu
```

### GPU Container

Build the GPU Docker image:
```
docker build -f Dockerfile.gpu -t tirex2-inference-gpu .
```

Run the GPU container:
```
docker run --rm --gpus 1 -p 8000:8000 \
  -e HF_TOKEN \
  -v tirex-hf-cache:/home/ubuntu/.cache/huggingface \
  -v tirex-triton-cache:/var/cache/triton \
  -v tirex-inductor-cache:/var/cache/torchinductor \
  tirex2-inference-gpu
```

## Development Setup

Running the server (and the tests) also downloads the gated weights, so set `HF_TOKEN` in your environment first (`export HF_TOKEN=hf_xxxxxxxx`) or run `huggingface-cli login`.

### Install Python dependencies:
```
pip install -r requirements.txt -r requirements-dev.txt
```

### Run the server:
```
cd inference
python -m app.main
```

### Run Tests:
Run while starting the server locally:
```
cd inference
pytest tests
```

Run tests against a running container:
```
cd inference
TEST_START_SERVER=0 TEST_PORT=8000 pytest tests -s
```

## License

TiRex is licensed under the [NXAI community license](../LICENSE).
