# 1. Base image: Python slim
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your app code
COPY api/app.py .

# 5. Expose the port Render expects (10000)
EXPOSE 10000

# 6. Launch with Gunicorn (production-grade server)
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:10000", "--timeout", "300"]
