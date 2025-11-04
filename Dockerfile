FROM ubuntu:22.04

USER root

RUN apt update && \
    apt install -y python3 python3-pip curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Cài Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install jupyter

# Tạo thư mục làm việc Jupyter
WORKDIR /home/code
CMD ["jupyter", "notebook", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
