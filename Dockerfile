FROM python:3.10-slim-bullseye

RUN python -m pip install -U pip

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD [ "python", "/app/app.py" ]
