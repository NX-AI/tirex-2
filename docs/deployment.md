# Deployment

TiRex-2 ships a Docker-based inference server that exposes the model over three APIs:

- **HTTP API** (FastAPI)
- **MQTT** (request/reply over MQTT v5)
- **MCP** (Model Context Protocol, for tools like Claude Desktop)

Source code for the inference server is in
[`inference/`](https://github.com/NX-AI/tirex-2/tree/main/inference) and this page documents its
deployment.

## Container images

Two container images are published:

- [`ghcr.io/nx-ai/tirex2-cpu`](https://ghcr.io/nx-ai/tirex2-cpu) — Linux image for `linux/amd64` and `linux/arm64`. Runs on Linux,
  macOS, or Windows via Docker Desktop's Linux container backend.
- [`ghcr.io/nx-ai/tirex2-gpu`](https://ghcr.io/nx-ai/tirex2-gpu) — CUDA Linux image for `linux/amd64`. Runs on Linux with the
  NVIDIA Container Toolkit, or on Windows via Docker Desktop's WSL2 backend with NVIDIA WSL
  GPU support.

### Running a container

To start a container from either image, run:

=== "CPU Image"

    ```bash
    docker run -it -p 8000:8000 ghcr.io/nx-ai/tirex2-cpu
    ```

=== "GPU Image"

    ```bash
    docker run -it --gpus 1 -p 8000:8000 ghcr.io/nx-ai/tirex2-gpu
    ```

???+ info "Warmup and compilation"

    Both images download the model and compile the **univariate** forecast path with
    `torch.compile` at startup
    (C++ on CPU, Triton on GPU) to enable fast inference. This can take up to 20 seconds. Changing
    context length or prediction horizon does not trigger recompilation.

    However, the first **multivariate** request requires a separate, one-time compilation.

??? tip "Caching model weights"

    The weights are not baked into the image — the container downloads them from Hugging Face on
    first use. The cache path differs by image:

    - **CPU image**: runs as `appuser` (UID 1000), home `/home/appuser`.
    - **GPU image**: runs as `ubuntu`, home `/home/ubuntu`.

    The cache lands at the default location (`~/.cache/huggingface`) under that user's home. Mount a
    volume there to avoid re-downloading:

    === "CPU Image"

        ```bash
        docker run -it -p 8000:8000 \
          -v tirex2-cache:/home/appuser/.cache/huggingface \
          ghcr.io/nx-ai/tirex2-cpu
        ```

    === "GPU Image"

        ```bash
        docker run -it --gpus 1 -p 8000:8000 \
          -v tirex2-cache:/home/ubuntu/.cache/huggingface \
          ghcr.io/nx-ai/tirex2-gpu
        ```

    Without this volume, a stopped-and-restarted container keeps the weights, but a **recreated**
    container re-downloads them. This includes every `docker compose up` after an edit and every
    image update.

    If you point the cache at your own mount instead, make sure it's writable by the container's user.

## HTTP API

- **Base URL:** `http://localhost:8000` (`/` returns 404).
- **Swagger docs:** [http://localhost:8000/docs](http://localhost:8000/docs).
- **Liveness probe:** [`GET /health`](http://localhost:8000/health), also used by Docker's `HEALTHCHECK`.

Send a list of series with each request, even if you only need one forecast. The server processes
requests in batches of up to 512 series and reduces the batch size if device memory runs out.
A single batch can still exceed available memory, so size your requests for your hardware.

???+ warning "Caution: no authentication"

    The HTTP API doesn't require authentication, so keep it on a trusted network and don't expose
    it directly to the internet.

### Univariate endpoints

Use `POST /univariate/forecast/mean` or `POST /univariate/forecast/quantiles` to forecast a batch of
one-dimensional series with no covariates.

???+ note "Two endpoints for different outputs"

    The `/mean` endpoints return a batch of median forecasts. The `/quantiles` endpoints return a
    batch of all 9 quantiles (10%, 20%, ..., 90%) for the same inputs.

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
```

???+ note "Covariates are ignored"

    This endpoint shape has no covariate fields. To condition on covariates, even for a
    univariate series, use the multivariate endpoints below with a one-row `target`.

### Multivariate endpoints

Use `POST /multivariate/forecast/mean` or `POST /multivariate/forecast/quantiles` to forecast a
batch of multivariate time series. Each item contains a multi-row `target` and optional
multi-row `past_covariates` and `future_covariates` fields:

```bash
# Multivariate (multi-target) series
curl -s -X POST "http://localhost:8000/multivariate/forecast/mean" \
  -H 'Content-Type: application/json' \
  -d '{
        "context": [{"target": [[1, 2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 40, 50, 60, 70, 80]]}],
        "prediction_length": 5
      }'

# Multivariate with past and future covariates
curl -s -X POST "http://localhost:8000/multivariate/forecast/mean" \
  -H 'Content-Type: application/json' \
  -d '{
        "context": [{
          "target": [[1, 2, 3, 4, 5, 6, 7, 8]],
          "past_covariates": [[1, 0, 0, 1, 0, 0, 1, 0]],
          "future_covariates": [[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]]
        }],
        "prediction_length": 5
      }'
```

???+ note "Covariate length requirements"

    `past_covariates` must have the same length as `target` (the context length).
    `future_covariates` must span at least the context plus the prediction horizon
    (`context_length + prediction_length`). Any extra trailing steps beyond that are ignored. A wrong length is
    rejected with a `500` error.

Batching multiple multivariate series works the same way, as a list under `context`. See
[inference/README.md](https://github.com/NX-AI/tirex-2/blob/main/inference/README.md) for the
full set of curl/Python examples, including batched multivariate-with-covariates requests.

### Python client

You can access the HTTP API from Python with the `requests` package:

```python
import requests

resp = requests.post(
    "http://localhost:8000/univariate/forecast/mean",
    json={"context": [[1, 2, 3, 4, 5, 6, 7, 8]], "prediction_length": 5},
)
print(resp.json())
```

## MQTT API

The MQTT integration uses **MQTT v5**. Send forecast requests to the request topics and set the
`Response Topic` property to tell TiRex-2 where to send the results. Each client can use its
own response topic, so there's no need for a shared results topic.

Requests must include a `Response Topic`. You can also set `Correlation Data` to match replies
to their requests.

You'll need a broker that supports **MQTT v5**. For a quick test, you can use
[broker.emqx.io](https://broker.emqx.io) with the [MQTTX CLI](https://mqttx.app/cli).
Don't send sensitive data through a public broker.

### Server configuration

Start TiRex-2 with MQTT enabled and connect it to the broker:

=== "CPU Image"

    ```bash
    docker run -p 8000:8000 -it \
      -e MQTT_ENABLED=1 \
      -e MQTT_BROKER_HOST=broker.emqx.io \
      -e MQTT_BROKER_PORT=1883 \
      ghcr.io/nx-ai/tirex2-cpu
    ```

=== "GPU Image"

    ```bash
    docker run --gpus 1 -p 8000:8000 -it \
      -e MQTT_ENABLED=1 \
      -e MQTT_BROKER_HOST=broker.emqx.io \
      -e MQTT_BROKER_PORT=1883 \
      ghcr.io/nx-ai/tirex2-gpu
    ```

### Client usage

In a new terminal, install the MQTTX CLI on Linux x86_64:

```bash
curl -sL https://github.com/emqx/MQTTX/releases/latest/download/mqttx-cli-linux-x64 -o mqttx \
  && sudo install mqttx /usr/local/bin/mqttx
```

Subscribe to your reply topic using MQTT v5:

```bash
mqttx sub -V 5 -t 'tirex/my-client/result' -h 'broker.emqx.io' -p 1883
```

Then, in a separate terminal, send a forecast request with `Response Topic` set to your reply topic:

```bash
mqttx pub -V 5 \
  -t 'tirex/univariate/forecast/request' \
  --response-topic 'tirex/my-client/result' \
  --correlation-data '1234' \
  -h 'broker.emqx.io' -p 1883 \
  -m '{"id": "1234", "context": [[0, 1, 2, 3]], "prediction_length": 4}'
```

The reply arrives on your chosen topic with the same `Correlation Data`. It contains `mean`
and `quantiles` if the forecast succeeds, or an `error` field if processing fails.

## MCP

Start the container as in [Running a container](#running-a-container), then connect a tool like Claude
Desktop by following its
[guide for connecting local servers](https://modelcontextprotocol.io/docs/develop/connect-local-servers).
Add the following `mcpServers` entry to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tirex": { "command": "npx", "args": ["-y", "mcp-remote", "http://127.0.0.1:8000/mcp"] }
  }
}
```

Two MCP tools are exposed:

- **Univariate:** `tirex_model(context, prediction_length)`.
- **Multivariate:**
  `tirex_model_multivariate(target, prediction_length, past_covariates, future_covariates)`.

Unlike the HTTP and MQTT APIs, MCP is **not batched** — each call forecasts a single series.

## Configuration options

Set environment variables with `-e`, for example:

=== "CPU Image"

    ```bash
    docker run -p 8000:8000 -e MODEL_DEVICE=cpu ghcr.io/nx-ai/tirex2-cpu
    ```

=== "GPU Image"

    ```bash
    docker run --gpus 1 -p 8000:8000 -e MODEL_DEVICE=cuda ghcr.io/nx-ai/tirex2-gpu
    ```

Available options are:

| Environment Variable               | Default Value                         | Description                                                                      |
| :--------------------------------- | :------------------------------------ | :------------------------------------------------------------------------------- |
| `MODEL_PATH`                       | `NX-AI/TiRex-2`                       | The Hugging Face model ID.                                                       |
| `MODEL_DEVICE`                     | `cpu` (CPU image), `cuda` (GPU image) | Device to run the model on (`cpu` or `cuda`).                                    |
| `HTTP_HOST`                        | `0.0.0.0`                             | Host the HTTP server binds to.                                                   |
| `HTTP_PORT`                        | `8000`                                | Port the HTTP server binds to.                                                   |
| `MQTT_ENABLED`                     | `0`                                   | Enable MQTT client functionality (`1` = true, `0` = false).                      |
| `MQTT_BROKER_HOST`                 | `None`                                | Hostname or IP address of the MQTT broker.                                       |
| `MQTT_BROKER_PORT`                 | `None`                                | Port of the MQTT broker.                                                         |
| `MQTT_BROKER_USERNAME`             | `None`                                | Username for authenticating with the MQTT broker (if required).                  |
| `MQTT_BROKER_PASSWORD`             | `None`                                | Password for authenticating with the MQTT broker (if required).                  |
| `MQTT_CLIENT_ID`                   | `tirex-worker`                        | Stable, unique client ID so the broker can resume the session on reconnect.      |
| `MQTT_SESSION_EXPIRY`              | `3600`                                | Seconds the broker retains the session (and queued requests) while disconnected. |
| `MQTT_TOPIC_UNIVARIATE_FORECAST`   | `tirex/univariate/forecast/request`   | Topic to subscribe to for univariate forecast requests.                          |
| `MQTT_TOPIC_MULTIVARIATE_FORECAST` | `tirex/multivariate/forecast/request` | Topic to subscribe to for multivariate forecast requests.                        |

## Building the images yourself

```bash
cd inference
```

=== "CPU Image"

    ```bash
    docker build -f Dockerfile.cpu -t tirex2-inference-cpu .
    docker run --rm -p 8000:8000 tirex2-inference-cpu
    ```

=== "GPU Image"

    ```bash
    docker build -f Dockerfile.gpu -t tirex2-inference-gpu .
    docker run --rm --gpus 1 -p 8000:8000 tirex2-inference-gpu
    ```

## Development setup

From the `inference/` directory, activate a virtual environment, install the requirements,
and start the server:

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m app.main
```

The test suite starts its own server by default:

```bash
pytest tests
```

Or against an running container:

```bash
TEST_START_SERVER=0 TEST_PORT=8000 pytest tests -s
```

## License

The inference server (this Docker image and the `inference/` directory) is licensed under the
same [Apache License 2.0](https://github.com/NX-AI/tirex-2/blob/main/LICENSE) as the rest of
TiRex-2.
