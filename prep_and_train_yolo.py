import os
import shutil
import random
from ultralytics import YOLO

def prepare_dataset():
    raw_dir = "dataset/raw_images"
    yolo_dir = "dataset_yolo"
    
    if not os.path.exists(raw_dir):
        print(f"Erro: A pasta {raw_dir} não existe!")
        print("Você precisa primeiro coletar as imagens rodando: python scripts/collect_yolo_images.py")
        import sys
        sys.exit(1)
    
    if os.path.exists(yolo_dir):
        shutil.rmtree(yolo_dir)
        
    train_dir = os.path.join(yolo_dir, "train")
    val_dir = os.path.join(yolo_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    split_ratio = 0.8
    
    for cls_name in os.listdir(raw_dir):
        cls_path = os.path.join(raw_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
            
        images = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        os.makedirs(os.path.join(train_dir, cls_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls_name), exist_ok=True)
        
        for img in train_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(train_dir, cls_name, img))
            
        for img in val_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(val_dir, cls_name, img))
            
    print("Dataset preparado em dataset_yolo/")

if __name__ == "__main__":
    import shutil
    
    print("Preparando dataset...")
    prepare_dataset()
    
    print("Iniciando treinamento...")
    # Usa um modelo nano de classificação, ideal para máquinas comuns e rápido.
    model = YOLO("yolov8n-cls.pt")
    
    # 5 épocas apenas para demonstrar criação dos gráficos (pode aumentar depois)
    results = model.train(data="dataset_yolo", epochs=5, imgsz=224, batch=16, project="runs", name="libras")
    
    # Pegar o diretório final do treinamento que o YOLO acabou de gerar
    run_dir = results.save_dir
    print(f"Treinamento concluído em: {run_dir}")
    
    print("Movendo gráficos e modelo treinado para pastas estáticas (assets/ e model/)...")
    os.makedirs("assets", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    
    # Move pesos
    pesos_path = os.path.join(run_dir, "weights", "best.pt")
    if os.path.exists(pesos_path):
        shutil.copy(pesos_path, os.path.join("model", "best.pt"))
        
    # Move gráficos
    graficos = ["results.png", "confusion_matrix_normalized.png"]
    for grafico in graficos:
        grafico_path = os.path.join(run_dir, grafico)
        if os.path.exists(grafico_path):
            shutil.copy(grafico_path, os.path.join("assets", grafico))
            
    print("Sucesso! Os itens já estão atualizados no caminho seguro. Você não precisa buscar neles em runs/... mais.")
