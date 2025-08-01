FROM python:3.10-slim

# Install ffmpeg and dependencies
RUN apt-get update && apt-get install -y ffmpeg libgl1 && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy app files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 10000

# Run your app (adjust if your main file or app name differs)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
