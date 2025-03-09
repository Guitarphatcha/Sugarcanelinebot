FROM python:3.10.12

# Copy files into /app
COPY . /app
WORKDIR /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Use the value of the PORT environment variable if available, fallback to 5000 if not
CMD uvicorn main:app --host=0.0.0.0 --port=${PORT:-5000}
