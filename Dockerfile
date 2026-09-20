FROM python:3.11-slim

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY main.py .
COPY model.py .
COPY use_labels.py .
COPY preprocessing.py .

# Copy the trained model artifacts (weights + label explanation)
COPY outputs/run_2026-09-20_19-43-46/ ./outputs/run_2026-09-20_19-43-46/

# Inform Docker of the listening port
EXPOSE 8000

# 4. Run the Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]