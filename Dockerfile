FROM python:3.13-slim

WORKDIR /app

# PyTorch CPU版（AlphaZero AI用）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 依存ライブラリ
RUN pip install --no-cache-dir \
    numpy \
    fastapi \
    "uvicorn[standard]"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
