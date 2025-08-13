FROM python:3.10.16-slim

RUN apt-get update --allow-unauthenticated  && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    redis-tools \
    libopencv-dev \
    tzdata\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /sdk

ENV PYTHONUNBUFFERED=1
ENV TZ=Europe/Moscow

COPY requirements.txt requirements.txt

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN pip install -e /app/.

CMD ["tail","-f","/dev/null"]