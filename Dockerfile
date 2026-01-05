# Dockerfile - install wkhtmltopdf via upstream .deb on python:3.11-slim
FROM python:3.11-slim

# Cache busting - change this to force rebuild
ARG CACHE_BUST=4

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime deps for WeasyPrint, wkhtmltopdf, and pyodbc (SQL Server)
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
      # ODBC driver dependencies for SQL Server
      unixodbc \
      unixodbc-dev \
      # WeasyPrint dependencies (complete set)
      libcairo2 \
      libpango-1.0-0 \
      libpangocairo-1.0-0 \
      libgdk-pixbuf-xlib-2.0-0 \
      libgdk-pixbuf-2.0-0 \
      libglib2.0-0 \
      libgobject-2.0-0 \
      libffi8 \
      libffi-dev \
      shared-mime-info \
      python3-cffi \
      pkg-config && \
      echo "Cache bust: ${CACHE_BUST}" && \
      # Install Microsoft ODBC Driver 18 for SQL Server
      curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
      echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/11/prod bullseye main" > /etc/apt/sources.list.d/mssql-release.list && \
      apt-get update && \
      ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 && \
      # Update library cache
      ldconfig && \
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
