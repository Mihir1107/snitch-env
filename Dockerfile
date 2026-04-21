FROM python:3.11.9-slim-bookworm

# HF Spaces runs as non-root — create user early
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# HF Spaces default port
ENV PORT=7860

# Switch to non-root
USER appuser

EXPOSE 7860

CMD ["server"]