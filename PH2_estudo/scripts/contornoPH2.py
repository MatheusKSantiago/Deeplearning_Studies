import cv2
import numpy as np
import os


dir_dataset_contornado = os.path.join("..","datasets","PH2Contornado")


def contornar(mask):
    
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mascara_filtrada = np.zeros_like(mask)

    for contorno in contornos:
        cv2.contourArea(contorno)
        cv2.drawContours(mascara_filtrada, [contorno], -1, 255, thickness=cv2.FILLED)
    return mascara_filtrada



def contornoImagem(img_path,img_seg_path,nomeImagem,formato):
    imagem_colorida = cv2.imread(img_path)
    mascara_cinza = cv2.imread(img_seg_path, cv2.IMREAD_GRAYSCALE)

    _, mascara_binaria = cv2.threshold(mascara_cinza, 127, 255, cv2.THRESH_BINARY)

    imagem_final_invertida = cv2.bitwise_and(imagem_colorida, imagem_colorida, mask=contornar(mascara_binaria))
    
    cv2.imwrite(os.path.join(dir_dataset_contornado,f"{nomeImagem}.{formato}"),imagem_final_invertida)



imagens_dir_path = os.path.join('..','datasets','PH2Dataset','PH2 Dataset images')
dirs = os.listdir(imagens_dir_path)

for img_dir in dirs:
    lista = os.listdir(f"{os.path.join(imagens_dir_path,img_dir)}")
    imagem_original_dir = lista[lista.index(f"{img_dir+'_Dermoscopic_Image'}")]
    imagem_seg_dir = lista[lista.index(f"{img_dir+'_lesion'}")]

    img_path = os.path.join(imagens_dir_path,img_dir,imagem_original_dir,f"{img_dir}.bmp")
    img_seg_path = os.path.join(imagens_dir_path,img_dir,imagem_seg_dir,f"{img_dir}_lesion.bmp")

    
    contornoImagem(img_path,img_seg_path,img_dir,"bmp")