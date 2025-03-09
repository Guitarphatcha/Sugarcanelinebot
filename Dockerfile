FROM python:3.10.12

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8080
CMD uvicorn main:app --host=0.0.0.0 --port=$PORT