import cv2
import time
import threading
import os
from ultralytics import YOLO

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(
    directory="templates"
)

# -------------------------
# ESTADO GLOBAL
# -------------------------

texto_atual = ""
mensagem_enviada = ""

letra_atual = "-"
confianca_atual = 0
tempo_restante = 0

TEMPO_CONFIRMACAO = 5

# -------------------------
# RECONHECIMENTO COM YOLO
# -------------------------

def iniciar_reconhecimento():

    global texto_atual
    global mensagem_enviada
    global letra_atual
    global confianca_atual
    global tempo_restante

    model_path = "model/best.pt"
    
    # Aguarda o modelo existir para não dar erro abrupto de inicialização
    # Isso ajuda caso o servidor rode antes do treinamento finalizar a exportação.
    while not os.path.exists(model_path):
        time.sleep(2)
        continue

    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)

    ultima_letra = None
    inicio_predicao = None

    while True:

        ret, frame = cap.read()

        if not ret:
            continue
            
        # Preditir a classe na imagem toda
        results = model.predict(frame, verbose=False)
        probs = results[0].probs

        if probs is not None:
            idx = probs.top1
            letra = results[0].names[idx]
            confianca = float(probs.top1conf.item())

            if confianca > 0.4:
                letra_atual = letra
                confianca_atual = confianca

                agora = time.time()

                if letra != ultima_letra:
                    ultima_letra = letra
                    inicio_predicao = agora
                else:
                    tempo_decorrido = (agora - inicio_predicao)
                    tempo_restante = max(0, TEMPO_CONFIRMACAO - tempo_decorrido)

                    if tempo_decorrido >= TEMPO_CONFIRMACAO:
                        if letra == "SPACE":
                            texto_atual += " "
                        elif letra == "DELETE":
                            texto_atual = texto_atual[:-1]
                        elif letra == "SEND":
                            mensagem_enviada = texto_atual
                            texto_atual = ""
                        else:
                            texto_atual += letra

                        # Reinicia para a proxima iteracao
                        inicio_predicao = agora
            else:
                letra_atual = "-"
                confianca_atual = confianca
                tempo_restante = 0
                ultima_letra = None
        else:
            letra_atual = "-"
            confianca_atual = 0
            tempo_restante = 0
            ultima_letra = None

        time.sleep(0.03)

# -------------------------
# ROTAS
# -------------------------

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )

@app.get("/status")
async def status():
    return JSONResponse(
        {
            "texto": texto_atual,
            "mensagem": mensagem_enviada,
            "letra": letra_atual,
            "confianca": confianca_atual,
            "restante": tempo_restante
        }
    )

# -------------------------
# STARTUP
# -------------------------

@app.on_event("startup")
async def startup():
    threading.Thread(
        target=iniciar_reconhecimento,
        daemon=True
    ).start()