FROM python:3.9

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script
COPY evaluate_model_bias.py .

# Set the entrypoint command
CMD ["python", "evaluate_model_bias.py"]
