# 1. Base image: Python slim
FROM python:3.10-slim

# 2. Install system libraries required by OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your Flask app
COPY app.py .

# 6. Expose the port Render expects
EXPOSE 10000

# 7. Launch with Gunicorn
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:10000", "--timeout", "300"]
