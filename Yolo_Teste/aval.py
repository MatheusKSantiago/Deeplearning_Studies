from ultralytics import YOLO

# Carregar o modelo
model = YOLO("runs\\classify\\train\weights\\best.pt")

# Avaliar o modelo com geração de matriz de confusão
results = model.val()