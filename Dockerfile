FROM python:3.9-slim

RUN pip install pipenv

WORKDIR /app

COPY ["", "", ""]

RUN pipenv install --system --deploy

COPY ["", "", ""]

EXPOSE 9696

ENTRYPOINT ["", "", "", ""]
# ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]
