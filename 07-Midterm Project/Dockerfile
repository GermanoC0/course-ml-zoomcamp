FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model.bin", "./"]

EXPOSE 9090
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9090", "predict:app"]
