# Imagen para Hugging Face Spaces (SDK: Docker). También sirve en Render o local:
#   docker build -t alertas . && docker run -p 8501:8501 alertas
FROM python:3.12-slim

# Hugging Face ejecuta el contenedor con el usuario 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 8501
# enableXsrfProtection=false: sin esto la carga de CSV falla (403) dentro del iframe de Spaces
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
