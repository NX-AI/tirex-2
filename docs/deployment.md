# Deployment

TiRex-2 ships a Docker-based inference server that exposes the model over three APIs:

- **HTTP API** (FastAPI)
- **MQTT** (request/reply over MQTT v5)
- **MCP** (Model Context Protocol, for tools like Claude Desktop)

Source: [`inference/`](https://github.com/NX-AI/tirex-2/tree/main/inference) in the repository
— this page documents what is actually implemented there.

## Images

Two container images are published:

- `ghcr.io/nx-ai/tirex2-cpu` — Linux image for `linux/amd64` and `linux/arm64`. Runs on Linux,
  macOS, or Windows via Docker Desktop's Linux container backend.
- `ghcr.io/nx-ai/tirex2-gpu` — CUDA Linux image for `linux/amd64`. Runs on Linux with the
  NVIDIA Container Toolkit, or on Windows via Docker Desktop's WSL2 backend with NVIDIA WSL
  GPU support.

Both images run a warmup forecast on startup so the model is `torch.compile`d (C++ kernels on
CPU, Triton on GPU) before the first real request; this download-and-warmup step can take up
to ~10-20 seconds.

### Run the CPU image

```bash
docker run -it -p 8000:8000 ghcr.io/nx-ai/tirex2-cpu
```

PowerShell:

```powershell
docker run -it -p 8000:8000 ghcr.io/nx-ai/tirex2-cpu
```

### Run the GPU image

```bash
docker run -it --gpus 1 -p 8000:8000 ghcr.io/nx-ai/tirex2-gpu
```

PowerShell:

```powershell
docker run -it --gpus 1 -p 8000:8000 ghcr.io/nx-ai/tirex2-gpu
```

Once running, the HTTP API is at `http://localhost:8000/`, with Swagger docs at
[http://localhost:8000/docs](http://localhost:8000/docs).

## HTTP API

Every request is batched — pass a list of series even for a single forecast. There is no
internal batching, so choose a batch size appropriate for your hardware; larger batches are
more efficient but too-large batches can cause out-of-memory errors.

### Univariate endpoints

`POST /univariate/forecast/mean` and `POST /univariate/forecast/quantiles` take a batch of
plain 1D series:

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

### Multivariate endpoints

`POST /multivariate/forecast/mean` and `POST /multivariate/forecast/quantiles` take a batch
of objects, each with a multi-row `target` and optional `future_covariates`:

```bash
# Multivariate (multi-target) series
curl -s -X POST "http://localhost:8000/multivariate/forecast/mean" \
  -H 'Content-Type: application/json' \
  -d '{
        "context": [{"target": [[1, 2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 40, 50, 60, 70, 80]]}],
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
```

Batching multiple multivariate series works the same way, as a list under `context`. See
[inference/README.md](https://github.com/NX-AI/tirex-2/blob/main/inference/README.md) for the
full set of curl/Python examples, including batched multivariate-with-covariates requests.

### `/quantiles` vs `/mean`

The `/mean` endpoints return the median forecast. The `/quantiles` endpoints return all 8
quantiles (10, 20, 30, 50, 60, 70, 80, 90%) for the same inputs.

### Python client example

```python
import requests

resp = requests.post(
    "http://localhost:8000/univariate/forecast/mean",
    json={"context": [[1, 2, 3, 4, 5, 6, 7, 8]], "prediction_length": 5},
)
print(resp.json())
```

## MQTT API

The MQTT integration uses **MQTT v5** with a request/reply pattern: TiRex-2 subscribes to
fixed forecast *request* topics and publishes each result back to the
**response topic the requester specifies on the request** (the MQTT v5 `Response Topic`
property). Every client receives only its own results — there is **no shared result topic**,
unlike a design where all clients read from one common response topic.

Requests without a Response Topic are rejected. Optionally set `Correlation Data` to match a
reply back to its request.

You need a **v5-capable** MQTT broker. For quick testing, a public broker like
[broker.emqx.io](https://broker.emqx.io) works (don't send sensitive data to a public
broker). The [MQTTX CLI](https://mqttx.app/cli) is convenient for testing:

```bash
# Linux x86_64 — standalone binary
curl -sL https://github.com/emqx/MQTTX/releases/latest/download/mqttx-cli-linux-x64 -o mqttx && sudo install mqttx /usr/local/bin/mqttx
```

Start the container with MQTT enabled:

```bash
docker run -p 8000:8000 -it -e MQTT_ENABLED=1 -e MQTT_BROKER_HOST=broker.emqx.io -e MQTT_BROKER_PORT=1883 ghcr.io/nx-ai/tirex2-cpu
```

Subscribe to your own reply topic first, over MQTT v5:

```bash
mqttx sub -V 5 -t 'tirex/my-client/result' -h 'broker.emqx.io' -p 1883
```

Then send a forecast request, pointing its Response Topic at that reply topic:

```bash
mqttx pub -V 5 \
  -t 'tirex/univariate/forecast/request' \
  --response-topic 'tirex/my-client/result' \
  --correlation-data '1234' \
  -h 'broker.emqx.io' -p 1883 \
  -m '{"id": "1234", "context": [[0, 1, 2, 3]], "prediction_length": 4}'
```

The result is published to your Response Topic, with the Correlation Data echoed back.
Successful results contain `mean` and `quantiles`; if an error occurs during processing, the
message published to the same Response Topic contains an `error` field instead.

## MCP

Start the container as in the HTTP API section above, then connect a tool like Claude
Desktop by following its
[guide for connecting local servers](https://modelcontextprotocol.io/docs/develop/connect-local-servers).
Add the following to `claude_desktop_config.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "tirex": { "command": "npx", "args": ["-y", "mcp-remote", "http://127.0.0.1:8000/mcp"] }
  }
}
```

Two MCP tools are exposed: a univariate `tirex_model(context, prediction_length)` and a
multivariate `tirex_model_multivariate(target, prediction_length, past_covariates,
future_covariates)`. Unlike the HTTP and MQTT APIs, MCP is **not batched** — each call
forecasts a single series.

## Configuration options

Set these as environment variables via `-e`, e.g.
`docker run -e MODEL_DEVICE=cuda ghcr.io/nx-ai/tirex2-cpu`:

| Environment Variable | Default Value | Description |
| :-------------------- | :------------- | :----------- |
| `MODEL_PATH` | `NX-AI/TiRex-2` | The Hugging Face model id. |
| `MODEL_DEVICE` | `cpu` | Device to run the model on (`cpu` or `cuda`). |
| `HTTP_HOST` | `0.0.0.0` | Host the HTTP server binds to. |
| `HTTP_PORT` | `8000` | Port the HTTP server binds to. |
| `MQTT_ENABLED` | `0` | Enable MQTT client functionality (`1`=True, `0`=False). |
| `MQTT_BROKER_HOST` | `None` | Hostname or IP address of the MQTT broker. |
| `MQTT_BROKER_PORT` | `None` | Port of the MQTT broker. |
| `MQTT_BROKER_USERNAME` | `None` | Username for authenticating with the MQTT broker (if required). |
| `MQTT_BROKER_PASSWORD` | `None` | Password for authenticating with the MQTT broker (if required). |
| `MQTT_CLIENT_ID` | `tirex-worker` | Stable, unique client id so the broker can resume the session on reconnect. |
| `MQTT_SESSION_EXPIRY` | `3600` | Seconds the broker retains the session (and queued requests) while disconnected. |
| `MQTT_TOPIC_UNIVARIATE_FORECAST` | `tirex/univariate/forecast/request` | Topic to subscribe to for univariate forecast requests. |
| `MQTT_TOPIC_MULTIVARIATE_FORECAST` | `tirex/multivariate/forecast/request` | Topic to subscribe to for multivariate forecast requests. |

## Building the images yourself

```bash
cd inference
docker build -f Dockerfile.cpu -t tirex2-inference-cpu .
docker run --rm -p 8000:8000 tirex2-inference-cpu
```

```bash
docker build -f Dockerfile.gpu -t tirex2-inference-gpu .
docker run --rm --gpus 1 -p 8000:8000 tirex2-inference-gpu
```

## Development setup

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m app.main
```

Run the tests against a locally-started server:

```bash
pytest tests
```

Or against an already-running container:

```bash
TEST_START_SERVER=0 TEST_PORT=8000 pytest tests -s
```

## License

The inference server (this Docker image and the `inference/` directory) is licensed under the
same [Apache License 2.0](https://github.com/NX-AI/tirex-2/blob/main/LICENSE) as the rest of
TiRex-2.
