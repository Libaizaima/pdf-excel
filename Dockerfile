# syntax=docker/dockerfile:1
FROM python:3.11-slim

# 1) 系统依赖：Ghostscript(给Camelot)、Java(给Tabula)、Tesseract+中文、Poppler(pdf2image)、OCRmyPDF
#    以及 opencv 运行时需要的 libgl、glib 等
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ghostscript \
      default-jre-headless \
      tesseract-ocr tesseract-ocr-chi-sim \
      poppler-utils \
      ocrmypdf \
      fonts-noto-cjk \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2) Python 依赖
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 3) 复制脚本与 Web 应用
COPY wechat_pdf2excel_ocr.py /app/wechat_pdf2excel_ocr.py
COPY app /app/app
COPY web /app/web

# 4) 容器内缺省编码与时区（可按需调整）
ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

EXPOSE 8003

# 5) 入口：启动 Web 服务（保留可覆盖方式进行 CLI 调用）
ENTRYPOINT ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8003"]
# 覆盖 ENTRYPOINT 使用 CLI：
# docker run --rm --entrypoint python -v "$PWD/input:/data/in" -v "$PWD/output:/data/out" image /app/wechat_pdf2excel_ocr.py /data/in/a.pdf -o /data/out/a.xlsx
