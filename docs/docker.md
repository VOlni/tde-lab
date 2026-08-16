# Running TDE Lab with Docker

One image serves both the GUI and the CLI — useful when the recipient has
Docker but no Python setup, or for running long sweeps on a server.

## Prerequisites: start Docker first

The `docker` CLI needs the Docker daemon running, or every command fails
with `Cannot connect to the Docker daemon`.

- **macOS / Windows**: install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  and launch it (macOS: `open -a Docker`, or start it from Applications).
  Wait for the whale icon to settle — first start takes ~30 s.
- **Linux**: install Docker Engine and start the service:
  `sudo systemctl start docker` (add `enable` to start on boot).

Verify it's up before building:

```bash
docker info >/dev/null && echo "Docker is running"
```

## Build

```bash
cd tde_lab
docker build -t tde-lab .
```

The first build downloads the base image and Python packages (a few
minutes); rebuilds after code changes are much faster.

## GUI

```bash
docker run --rm -p 8501:8501 tde-lab
# open http://localhost:8501
```

Keep saved outputs on the host by mounting a volume:

```bash
docker run --rm -p 8501:8501 -v "$PWD/output:/app/output" tde-lab
```

## CLI

Any `tde` command works by overriding the default command:

```bash
# quick sweep, results appear in ./output on the host
docker run --rm -v "$PWD/output:/app/output" tde-lab \
    tde sweep-sas --quick -m standard,dist-l1

# long sweep with resume: also mount the chunk cache
docker run --rm -v "$PWD/output:/app/output" -v "$PWD/.cache:/app/.cache" tde-lab \
    tde sweep-sas -m standard,dist-l1 --realizations 10000 --seed 42

# real recording: mount the folder with the WAV files read-only
docker run --rm -v "$PWD/Signals:/data:ro" -v "$PWD/output:/app/output" tde-lab \
    tde wav "/data/50 см 1 м центр.wav" --mic-distance 0.5 --start 1 --duration 3
```

## Sharing the image itself

```bash
docker save tde-lab | gzip > tde-lab.tar.gz      # ~0.5 GB; email/drive-able
docker load < tde-lab.tar.gz                     # on the receiving machine
```

or push to a registry (`docker tag tde-lab ghcr.io/<user>/tde-lab:1.1.0 &&
docker push …`) so the recipient can just `docker run`.

## Notes

- The image is based on `python:3.12-slim`; `soundfile` wheels bundle
  libsndfile, so no extra system packages are needed.
- The container writes matplotlib caches to `/tmp` and outputs to
  `/app/output` — mount the latter to keep results.
- GUI resource usage is whatever you give the Docker VM; unlike Streamlit
  Cloud there is no 1 GB limit, so full sweeps are fine in the container.
