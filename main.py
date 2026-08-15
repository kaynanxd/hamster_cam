import os
import cv2
import numpy as np

import mediapipe as mp

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

# IDs das pontas e articulações intermediárias dos dedos
TIP_IDS = [4, 8, 12, 16, 20]     # Polegar, Indicador, Médio, Anelar, Mindinho
PIP_IDS = [2, 6, 10, 14, 18]

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
IMAGENS_GESTOS = {}

def carregar_imagens_customizadas():
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR, exist_ok=True)

    gestos_nomes = {
        # Gestos de 1 mão
        "PUNHO_FECHADO": "palmafechada",
        "APONTANDO": "apontandoneutro",
        "PAZ_E_AMOR": "paz",
        "PALMA_ABERTA": "palmaaberta",
        "POSITIVO": "like",
        "NEGATIVO": "dislike",
        
        # Gestos de 2 mãos
        "DUPLO_POSITIVO": "duplolike",
        "DUPLO_NEGATIVO": "duplodislike",
        "DUPLA_PALMA_ABERTA": "2maosabertas",
        "DUPLO_PUNHO_FECHADO": "palmafechadadupla",
        "DUPLO_PAZ_E_AMOR": "duplo_paz_e_amor",
        "PONTAS_SE_TOCANDO": "pensando",
        "CORACAO": "coracao",

        # Gestos / Reações com a Boca
        "BOCA_ABERTA": "bocaaberta",
        "BIQUINHO": "biquinho",
        "BIQUINHO_E_PAZ": "biquinhopaz",
        "SORRISO": "feliz",
        "SORRISO_APONTANDO": "apontandoesquerda",
        "LINGUA_FORA": "linguafora",
        "SURPRESA_GRITO": "surpresa",
        "EMPOLGADO": "empolgado",
        "RAIVA_GRITO": "pirulito",
        "DUPLO_PUNHO_BOCA_ABERTA": "supresagrito"
    }

    extensoes = [".jpg", ".png", ".jpeg", ".webp"]

    for gesto, nome_base in gestos_nomes.items():
        for ext in extensoes:
            filepath = os.path.join(ASSETS_DIR, nome_base + ext)
            if os.path.exists(filepath):
                img = cv2.imread(filepath)
                if img is not None:
                    IMAGENS_GESTOS[gesto] = cv2.resize(img, (400, 400))
                    break 

carregar_imagens_customizadas()

def contar_dedos(landmarks, hand_label):
    """
    Retorna uma lista de 5 booleanos indicando quais dedos estão levantados.
    [Polegar, Indicador, Médio, Anelar, Mindinho]
    """
    dedos = []

    # 1. Lógica do Polegar (baseada no eixo X dependendo da mão)
    if hand_label == "Right":
        dedos.append(landmarks[TIP_IDS[0]].x < landmarks[PIP_IDS[0]].x)
    else:
        dedos.append(landmarks[TIP_IDS[0]].x > landmarks[PIP_IDS[0]].x)

    # 2. Lógica para Indicador, Médio, Anelar e Mindinho (eixo Y)
    for i in range(1, 5):
        if landmarks[TIP_IDS[i]].y < landmarks[PIP_IDS[i]].y:
            dedos.append(True)
        else:
            dedos.append(False)

    return dedos

def classificar_gesto_individual(dedos, landmarks):
    """Traduz o estado dos dedos de UMA mão em um gesto individual."""
    total_levantados = dedos.count(True)

    if total_levantados == 0:
        return "PUNHO_FECHADO"
    if dedos == [False, True, False, False, False]:
        return "APONTANDO"
    if dedos == [False, True, True, False, False]:
        return "PAZ_E_AMOR"
    if total_levantados == 5:
        return "PALMA_ABERTA"

    if dedos == [True, False, False, False, False]:
        if landmarks[TIP_IDS[0]].y < landmarks[PIP_IDS[0]].y:
            return "POSITIVO"
        else:
            return "NEGATIVO"
    
    return "DESCONHECIDO"

def classificar_gesto_combinado(gestos_maos, hand_landmarks_list=None):
    qtd = len(gestos_maos)
    if qtd == 0:
        return "DESCONHECIDO"

    # Verificações geométricas com 2 mãos
    if hand_landmarks_list is not None and len(hand_landmarks_list) == 2:
        h1 = hand_landmarks_list[0].landmark
        h2 = hand_landmarks_list[1].landmark

        dist_ind = np.sqrt((h1[8].x - h2[8].x)**2 + (h1[8].y - h2[8].y)**2)
        dist_pol = np.sqrt((h1[4].x - h2[4].x)**2 + (h1[4].y - h2[4].y)**2)
        dist_med = np.sqrt((h1[12].x - h2[12].x)**2 + (h1[12].y - h2[12].y)**2)

        # 1. Coração Exige polegares juntos embaixo e indicadores no topo
        if dist_ind < 0.11 and dist_pol < 0.11:
            avg_ind_y = (h1[8].y + h2[8].y) / 2.0
            avg_pol_y = (h1[4].y + h2[4].y) / 2.0
            if avg_ind_y < avg_pol_y:
                return "CORACAO"

        # 2. Pontas se tocando Indicadores e médios juntos, MAS polegares NÃO estão colados embaixo
        if dist_ind < 0.09 and dist_med < 0.09 and dist_pol > 0.10:
            return "PONTAS_SE_TOCANDO"

    if qtd == 1:
        return gestos_maos[0]

    g1, g2 = gestos_maos[0], gestos_maos[1]

    if g1 == "POSITIVO" and g2 == "POSITIVO":
        return "DUPLO_POSITIVO"
    if g1 == "NEGATIVO" and g2 == "NEGATIVO":
        return "DUPLO_NEGATIVO"
    if g1 == "PALMA_ABERTA" and g2 == "PALMA_ABERTA":
        return "DUPLA_PALMA_ABERTA"
    if g1 == "PUNHO_FECHADO" and g2 == "PUNHO_FECHADO":
        return "DUPLO_PUNHO_FECHADO"
    if g1 == "PAZ_E_AMOR" and g2 == "PAZ_E_AMOR":
        return "DUPLO_PAZ_E_AMOR"

    return f"{g1} + {g2}"

def classificar_gesto_geral(gestos_maos, estado_boca, hand_landmarks_list=None):
    """Combina o gesto das mãos com o estado da boca"""
    gesto_maos = classificar_gesto_combinado(gestos_maos, hand_landmarks_list)

    if estado_boca == "ABERTA":
        if gesto_maos == "DUPLA_PALMA_ABERTA":
            return "SURPRESA_GRITO"           # Duas mãos abertas + boca aberta
        elif gesto_maos in ["POSITIVO", "DUPLO_POSITIVO"]:
            return "EMPOLGADO"                # Joinha + boca aberta
        elif gesto_maos == "PUNHO_FECHADO":
            return "RAIVA_GRITO"              # mão fechada + boca aberta
        elif gesto_maos == "DUPLO_PUNHO_FECHADO":
            return "DUPLO_PUNHO_BOCA_ABERTA"  # 2 mãos fechadas + boca aberta
        elif gesto_maos == "DESCONHECIDO":
            return "BOCA_ABERTA"              # Apenas boca aberta
        else:
            return gesto_maos                 # Sem combinação mapeada, Prioriza mão

    elif estado_boca == "BIQUINHO":
        if gesto_maos == "PAZ_E_AMOR":
            return "BIQUINHO_E_PAZ"    
        elif gesto_maos == "DESCONHECIDO":
            return "BIQUINHO"          
        else:
            return gesto_maos         

    elif estado_boca == "LINGUA_FORA":
        if gesto_maos == "DESCONHECIDO":
            return "LINGUA_FORA"      
        else:
            return gesto_maos       

    elif estado_boca == "SORRISO":
        if gesto_maos == "APONTANDO":
            return "SORRISO_APONTANDO"
        elif gesto_maos == "DESCONHECIDO":
            return "SORRISO"         
        else:
            return gesto_maos         

    return gesto_maos

def obter_imagem_exibicao(gesto):
    if gesto in IMAGENS_GESTOS:
        return IMAGENS_GESTOS[gesto]
    
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cores = {
        # Gestos de 1 mão
        "PUNHO_FECHADO": ((0, 0, 255), "Punho Fechado"),
        "APONTANDO": ((255, 165, 0), "Indicador"),
        "PAZ_E_AMOR": ((0, 255, 0), "paz e amor"),
        "PALMA_ABERTA": ((255, 255, 0), "Palma Aberta"),
        "POSITIVO": ((0, 255, 255), "Like"),
        "NEGATIVO": ((0, 0, 200), "Deslike"),
        
        # Gestos de 2 mãos
        "DUPLO_POSITIVO": ((0, 255, 255), "Duplo Like"),
        "DUPLO_NEGATIVO": ((0, 0, 200), "Duplo Deslike"),
        "DUPLA_PALMA_ABERTA": ((255, 255, 0), "Duas Palmas Abertas"),
        "DUPLO_PUNHO_FECHADO": ((0, 0, 255), "Duplo Punho Fechado"),
        "DUPLO_PAZ_E_AMOR": ((0, 255, 0), "Duplo Paz e Amor"),
        "PONTAS_SE_TOCANDO": ((180, 105, 255), "Pensando"),
        "CORACAO": ((147, 20, 255), "Coracao"),
        
        # Reações de Boca
        "BOCA_ABERTA": ((255, 105, 180), "Boca Aberta"),
        "BIQUINHO": ((255, 192, 203), "Biquinho"),
        "BIQUINHO_E_PAZ": ((255, 105, 180), "Selfie"),
        "SORRISO": ((0, 255, 255), "Sorriso"),
        "SORRISO_APONTANDO": ((0, 255, 255), "Sorriso + Apontando"),
        "LINGUA_FORA": ((0, 165, 255), "Lingua para fora"),
        "SURPRESA_GRITO": ((255, 0, 128), "SURPRESA / GRITO"),
        "EMPOLGADO": ((0, 255, 128), "EMPOLGADO"),
        "RAIVA_GRITO": ((0, 0, 180), "RAIVA / GRITO"),
        "DUPLO_PUNHO_BOCA_ABERTA": ((0, 0, 220), "2 Mãos Fechadas + Boca Aberta"),

        "DESCONHECIDO": ((80, 80, 80), "Nenhum Gesto")
    }
    
    cor, texto = cores.get(gesto, ((120, 60, 180), gesto))
    img[:] = cor

    tamanho_fonte = 0.7 if len(texto) > 15 else 1.0
    cv2.putText(img, texto, (15, 200), cv2.FONT_HERSHEY_SIMPLEX, tamanho_fonte, (255, 255, 255), 2)
    return img

# Cria janela redimensionável
cv2.namedWindow("Webcam - Reconhecimento de Gestos", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam - Reconhecimento de Gestos", 1024, 768)

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Converte BGR para RGB para processamento no MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Processa Mãos e Rosto
    resultado_hands = hands.process(rgb_frame)
    resultado_face = face_mesh.process(rgb_frame)

    gestos_detectados = []
    lista_landmarks_maos = []
    estado_boca = "FECHADA"

    # 1. Rastreamento da Boca (Face Mesh)
    if resultado_face.multi_face_landmarks:
        for face_landmarks in resultado_face.multi_face_landmarks:
            pt13 = face_landmarks.landmark[13]
            pt14 = face_landmarks.landmark[14]
            pt61 = face_landmarks.landmark[61]
            pt291 = face_landmarks.landmark[291]
            pt234 = face_landmarks.landmark[234]
            pt454 = face_landmarks.landmark[454]
            pt0 = face_landmarks.landmark[0]
            pt17 = face_landmarks.landmark[17]

            dist_vert = np.sqrt((pt13.x - pt14.x)**2 + (pt13.y - pt14.y)**2)
            dist_horiz = np.sqrt((pt61.x - pt291.x)**2 + (pt61.y - pt291.y)**2)
            dist_rosto = np.sqrt((pt234.x - pt454.x)**2 + (pt234.y - pt454.y)**2)
            dist_labios_ext = np.sqrt((pt0.x - pt17.x)**2 + (pt0.y - pt17.y)**2)

            if dist_horiz > 0 and dist_rosto > 0:
                mar = dist_vert / dist_horiz   
                mwr = dist_horiz / dist_rosto  
                razao_labios = dist_vert / dist_labios_ext if dist_labios_ext > 0 else 0

                cantos_avg_y = (pt61.y + pt291.y) / 2.0
                elevacao_cantos = pt14.y - cantos_avg_y

                if mar > 0.45:
                    estado_boca = "ABERTA"
                elif mwr < 0.24:
                    estado_boca = "BIQUINHO"
                elif mwr > 0.40 and elevacao_cantos > 0.015 and mar < 0.25:
                    estado_boca = "SORRISO"
                elif mar >= 0.16 and mar <= 0.45 and (pt14.y - cantos_avg_y) > 0.010:
                    estado_boca = "LINGUA_FORA"

    # 2. Rastreamento das Mãos (Hands)
    if resultado_hands.multi_hand_landmarks and resultado_hands.multi_handedness:
        for hand_landmarks, handedness in zip(resultado_hands.multi_hand_landmarks, resultado_hands.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            tipo_mao = handedness.classification[0].label
            dedos_estado = contar_dedos(hand_landmarks.landmark, tipo_mao)
            gesto_ind = classificar_gesto_individual(dedos_estado, hand_landmarks.landmark)
            gestos_detectados.append(gesto_ind)
            lista_landmarks_maos.append(hand_landmarks)

    # 3. Combinação Gesto Mãos + Estado da Boca
    gesto_final = classificar_gesto_geral(gestos_detectados, estado_boca, lista_landmarks_maos)

    # Exibe a imagem de resposta apenas se um gesto com imagem customizada for detectado
    if gesto_final in IMAGENS_GESTOS:
        img_resultado = IMAGENS_GESTOS[gesto_final]
        overlay_w, overlay_h = 220, 220
        if h > overlay_h + 30 and w > overlay_w + 30:
            img_resized = cv2.resize(img_resultado, (overlay_w, overlay_h))
            x1 = w - overlay_w - 20
            y1 = 20
            x2 = w - 20
            y2 = 20 + overlay_h

            frame[y1:y2, x1:x2] = img_resized

    cv2.imshow("Webcam - Reconhecimento de Gestos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Webcam - Reconhecimento de Gestos", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()