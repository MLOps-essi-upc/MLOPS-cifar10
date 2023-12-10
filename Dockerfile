FROM python:3.9

EXPOSE 8000

WORKDIR /app

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY ./models /app/models 

COPY ./src .

COPY ./static /app/static 

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "app.api:app"]
