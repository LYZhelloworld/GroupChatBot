# Container configuration
FROM library/python:3.12.10-slim-bullseye
WORKDIR /app
COPY . .

# Set Python index URL if you cannot download from official source
# ARG PYTHON_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

# Set environment variables
RUN pip config set global.index-url "${PYTHON_INDEX}"
ENV UV_INDEX_URL=${PYTHON_INDEX}

# Build
RUN pip install uv
RUN uv venv
RUN uv sync

# Run
CMD ["uv", "run", "main.py"]