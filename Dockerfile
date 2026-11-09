FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY strategy_finder/ strategy_finder/
RUN pip install --no-cache-dir .
ENTRYPOINT ["strategy-finder"]
