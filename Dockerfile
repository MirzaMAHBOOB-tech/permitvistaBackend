# Dockerfile - install wkhtmltopdf via upstream .deb on python:3.11-slim
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime deps for WeasyPrint and wkhtmltopdf
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      fonts-dejavu-core \
      libxrender1 \
      libxext6 \
      libfontconfig1 \
      libssl3 \
      gnupg \
      dirmngr \
      # WeasyPrint dependencies
      libcairo2 \
      libpango-1.0-0 \
      libpangocairo-1.0-0 \
      libgdk-pixbuf2.0-0 \
      libglib2.0-0 \
      libgobject-2.0-0 \
      libffi-dev \
      shared-mime-info \
      python3-cffi && \
    # Download wkhtmltopdf prebuilt deb (debian11 build usually works on slim images)
    curl -L -o /tmp/wkhtml.deb \
      "https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.debian11_amd64.deb" && \
    # Install and fix missing deps if any
    dpkg -i /tmp/wkhtml.deb || apt-get -y -f install && \
    rm -f /tmp/wkhtml.deb && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Render sets $PORT; bind to 0.0.0.0
CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
