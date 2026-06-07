import cv2
import os

LETRA = input("Digite a letra ou classe que deseja coletar (ex: A, B, DELETE, ESPACO): ").strip().upper()
pasta_destino = f"dataset/raw_images/{LETRA}"

os.makedirs(pasta_destino, exist_ok=True)

contador = len([f for f in os.listdir(pasta_destino) if f.endswith('.jpg')])

cap = cv2.VideoCapture(0)

print(f"\n--- Coletando imagens para a classe: {LETRA} ---")
print(f"Imagens atuais: {contador}")
print("Pressione 'S' para salvar um frame (imagem).")
print("Pressione 'ESC' para sair.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f"Classe: {LETRA}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Amostras: {contador}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Pressione 'S' para salvar", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Coleta Dataset YOLO", frame)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27: # ESC
        break
    elif tecla == ord('s'):
        caminho_imagem = os.path.join(pasta_destino, f"{contador}.jpg")
        cv2.imwrite(caminho_imagem, frame)
        contador += 1
        print(f"Imagem salva: {caminho_imagem}")

cap.release()
cv2.destroyAllWindows()
