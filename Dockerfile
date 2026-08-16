FROM python:3.12-slim

WORKDIR /app

# soundfile>=0.12 wheels bundle libsndfile — no system packages needed
COPY pyproject.toml README.md LICENSE streamlit_app.py ./
COPY src ./src
RUN pip install --no-cache-dir ".[gui]"

# writable matplotlib cache for arbitrary run-time users
ENV MPLCONFIGDIR=/tmp/mpl

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# default: the GUI; override the command to use the CLI, e.g.
#   docker run --rm -v "$PWD/output:/app/output" tde-lab tde sweep-sas --quick
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
