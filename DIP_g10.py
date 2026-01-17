import cv2
import numpy as np
import math
import mediapipe as mp
import pygame

# ---------------------------
# Configuración Inicial
# ---------------------------
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Índices MediaPipe Hands
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
INDEX_MCP = 5

# ---------------------------
# Funciones Matemáticas
# ---------------------------
def calcular_angulo(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    v1 = np.array([x1 - x2, y1 - y2])
    v2 = np.array([x3 - x2, y3 - y2])
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0.0
    cos_ang = dot / (norm1 * norm2)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return math.degrees(math.acos(cos_ang))

def calcular_angulo_munieca(wrist, ref_point):
    x1, y1 = wrist
    x2, y2 = ref_point
    vx = x2 - x1
    vy = y1 - y2  # Invertimos Y
    return math.degrees(math.atan2(vy, vx))

def distancia_normalizada(p1, p2, ref_len):
    d = math.dist(p1, p2)
    if ref_len == 0: return 0.0
    return d / ref_len

# ---------------------------
# Pygame: Inicialización y Dibujo
# ---------------------------
pygame.init()
WINDOW_W, WINDOW_H = 960, 540
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("Sistema de Teleoperación Robótica")
clock = pygame.time.Clock()

# Fuentes
font_big = pygame.font.SysFont("Arial", 40, bold=True)
font_small = pygame.font.SysFont("Arial", 18)

# Colores
BG_COLOR = (8, 10, 25)
BTN_COLOR = (40, 120, 220)
BTN_HOVER = (60, 140, 250)
TXT_COLOR = (255, 255, 255)

# Variables Globales del Robot (Persistencia)
robot_state = {
    "codo": 90.0,
    "hombro": math.radians(90),
    "munieca": 0.0,
    "apertura": 0.5, # 50%
    "detected_pose": False,
    "detected_hand": False
}

# ---------------------------
# Funciones de Dibujo Robot
# ---------------------------
def draw_link(surface, p1, p2, width, color):
    pygame.draw.line(surface, color, p1, p2, width)
    pygame.draw.circle(surface, (60, 80, 140), p1, width//2)
    pygame.draw.circle(surface, (60, 80, 140), p2, width//2)

def draw_gripper(surface, base, ang_rad, apertura):
    cx, cy = base
    L_dedo = 40
    sep_max = math.radians(40)
    sep = apertura * sep_max
    
    # Dedo 1
    a1 = ang_rad + sep/2
    x1 = cx + L_dedo * math.cos(a1)
    y1 = cy - L_dedo * math.sin(a1) # Y invertida en pantalla
    pygame.draw.line(surface, (0, 200, 255), (cx, cy), (x1, y1), 8)
    
    # Dedo 2
    a2 = ang_rad - sep/2
    x2 = cx + L_dedo * math.cos(a2)
    y2 = cy - L_dedo * math.sin(a2)
    pygame.draw.line(surface, (0, 200, 255), (cx, cy), (x2, y2), 8)

# ---------------------------
# Estados del Programa
# ---------------------------
STATE_MENU = 0
STATE_APP = 1
current_state = STATE_MENU
selected_side = None # "RIGHT" o "LEFT"

# ---------------------------
# Bucle Principal
# ---------------------------
cap = cv2.VideoCapture(0)

# Inicializar modelos
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) 
# Nota: max_num_hands=2 para detectar ambas y filtrar la correcta por distancia

running = True
while running:
    # 1. Gestión de Eventos
    mouse_pos = pygame.mouse.get_pos()
    click = False
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if current_state == STATE_APP:
                current_state = STATE_MENU # Volver al menú
            else:
                running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: click = True

    screen.fill(BG_COLOR)

    # ==========================================================
    # LÓGICA: MENÚ DE SELECCIÓN
    # ==========================================================
    if current_state == STATE_MENU:
        # Título
        title = font_big.render("Seleccione el Brazo a Utilizar", True, TXT_COLOR)
        screen.blit(title, (WINDOW_W//2 - title.get_width()//2, 100))

        # Botón Izquierda
        btn_l_rect = pygame.Rect(WINDOW_W//2 - 250, 250, 200, 80)
        col_l = BTN_HOVER if btn_l_rect.collidepoint(mouse_pos) else BTN_COLOR
        pygame.draw.rect(screen, col_l, btn_l_rect, border_radius=10)
        txt_l = font_small.render("Brazo IZQUIERDO", True, TXT_COLOR)
        screen.blit(txt_l, (btn_l_rect.centerx - txt_l.get_width()//2, btn_l_rect.centery - txt_l.get_height()//2))

        # Botón Derecha
        btn_r_rect = pygame.Rect(WINDOW_W//2 + 50, 250, 200, 80)
        col_r = BTN_HOVER if btn_r_rect.collidepoint(mouse_pos) else BTN_COLOR
        pygame.draw.rect(screen, col_r, btn_r_rect, border_radius=10)
        txt_r = font_small.render("Brazo DERECHO", True, TXT_COLOR)
        screen.blit(txt_r, (btn_r_rect.centerx - txt_r.get_width()//2, btn_r_rect.centery - txt_r.get_height()//2))

        # Lógica de Click
        if click:
            if btn_l_rect.collidepoint(mouse_pos):
                selected_side = "LEFT"
                current_state = STATE_APP
            elif btn_r_rect.collidepoint(mouse_pos):
                selected_side = "RIGHT"
                current_state = STATE_APP

    # ==========================================================
    # LÓGICA: APLICACIÓN PRINCIPAL
    # ==========================================================
    elif current_state == STATE_APP:
        ret, frame = cap.read()
        if not ret: break

        # Espejo (Flip Horizontal)
        frame = cv2.flip(frame, 1)
        h_cam, w_cam, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesamiento MediaPipe
        res_pose = pose.process(img_rgb)
        res_hands = hands.process(img_rgb)

        robot_state["detected_pose"] = False
        robot_state["detected_hand"] = False
        
        # Coordenada de la muñeca del brazo (para filtrar manos lejanas)
        wrist_arm_pos = None 

        # --- 1. PROCESAR POSE (Brazo) ---
        if res_pose.pose_landmarks:
            lm = res_pose.pose_landmarks.landmark
            
            # Selección de landmarks según lado
            if selected_side == "RIGHT":
                # Si el usuario eligió derecha, y la cámara es espejo:
                # La 'derecha' del usuario aparece a la derecha de la pantalla.
                # MediaPipe Pose detecta 'RIGHT' correctamente en el cuerpo de la persona.
                idx_s = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                idx_e = mp_pose.PoseLandmark.RIGHT_ELBOW.value
                idx_w = mp_pose.PoseLandmark.RIGHT_WRIST.value
            else:
                idx_s = mp_pose.PoseLandmark.LEFT_SHOULDER.value
                idx_e = mp_pose.PoseLandmark.LEFT_ELBOW.value
                idx_w = mp_pose.PoseLandmark.LEFT_WRIST.value

            s = lm[idx_s]
            e = lm[idx_e]
            w = lm[idx_w]

            # Filtro de visibilidad
            if s.visibility > 0.5 and e.visibility > 0.5 and w.visibility > 0.5:
                robot_state["detected_pose"] = True
                
                # Coordenadas pixel
                p_s = (int(s.x * w_cam), int(s.y * h_cam))
                p_e = (int(e.x * w_cam), int(e.y * h_cam))
                p_w = (int(w.x * w_cam), int(w.y * h_cam))
                wrist_arm_pos = p_w # Guardamos posición muñeca brazo

                # Dibujar esqueleto en cámara
                cv2.line(frame, p_s, p_e, (0, 255, 0), 4)
                cv2.line(frame, p_e, p_w, (0, 255, 0), 4)
                cv2.circle(frame, p_s, 5, (255, 0, 0), -1)
                
                # Cálculos
                ang_codo = calcular_angulo(p_s, p_e, p_w)
                # Flexión: si el brazo está recto es 180, flexión es 180 - ángulo
                flex = 180 - ang_codo
                robot_state["codo"] = max(0, min(180, flex))

                # Ángulo hombro (visual)
                v_se = np.array([p_e[0] - p_s[0], p_s[1] - p_e[1]])
                if np.linalg.norm(v_se) != 0:
                    robot_state["hombro"] = math.atan2(v_se[1], v_se[0])

        # --- 2. PROCESAR MANOS (Con filtro de distancia) ---
        if res_hands.multi_hand_landmarks and wrist_arm_pos is not None:
            min_dist = 10000
            best_hand_lm = None

            # Iterar sobre TODAS las manos detectadas
            for hand_landmarks in res_hands.multi_hand_landmarks:
                # Obtener la muñeca de esta mano candidata
                w_h = hand_landmarks.landmark[WRIST]
                px, py = int(w_h.x * w_cam), int(w_h.y * h_cam)
                
                # Calcular distancia a la muñeca del brazo detectado
                dist = math.dist((px, py), wrist_arm_pos)
                
                # Si la mano está cerca de la muñeca del brazo (< 60 px aprox), es la correcta
                if dist < 80: # Umbral de tolerancia
                    if dist < min_dist:
                        min_dist = dist
                        best_hand_lm = hand_landmarks

            # Si encontramos una mano válida asociada al brazo
            if best_hand_lm:
                robot_state["detected_hand"] = True
                
                # Extraer puntos
                pts = []
                for l in best_hand_lm.landmark:
                    pts.append((int(l.x * w_cam), int(l.y * h_cam)))
                
                p_wrist = pts[WRIST]
                p_idx = pts[INDEX_MCP]
                p_idx_tip = pts[INDEX_TIP]
                p_thumb = pts[THUMB_TIP]

                # Dibujar mano correcta
                cv2.line(frame, p_wrist, p_idx, (255, 0, 255), 2)
                cv2.circle(frame, p_thumb, 5, (0, 255, 255), -1)
                cv2.circle(frame, p_idx_tip, 5, (0, 255, 255), -1)

                # Cálculos Gripper
                robot_state["munieca"] = calcular_angulo_munieca(p_wrist, p_idx)
                
                d_ref = math.dist(p_wrist, p_idx)
                d_fingers = math.dist(p_thumb, p_idx_tip)
                
                apertura = 0
                if d_ref > 0:
                    val = d_fingers / d_ref
                    # Calibración empírica
                    val = (val - 0.2) / (0.7 - 0.2)
                    apertura = max(0.0, min(1.0, val))
                
                robot_state["apertura"] = apertura

        # --- RENDERIZADO (Dividir Pantalla) ---
        # Izquierda: Cámara
        frame_rgb_view = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb_view = np.rot90(frame_rgb_view)
        surf_cam = pygame.surfarray.make_surface(frame_rgb_view)
        surf_cam = pygame.transform.smoothscale(surf_cam, (WINDOW_W//2, WINDOW_H))
        screen.blit(surf_cam, (0, 0))

        # Derecha: Robot
        offset_x = WINDOW_W // 2
        pygame.draw.rect(screen, (15, 20, 40), (offset_x, 0, offset_x, WINDOW_H))
        
        # Grid visual
        for i in range(0, WINDOW_H, 40):
            pygame.draw.line(screen, (25, 35, 60), (offset_x, i), (WINDOW_W, i))

        # Cinemática Robot
        base_robot = (offset_x + WINDOW_W//4, WINDOW_H - 100)
        L_brazo = 140
        
        # Eslabón 1
        ang_h = robot_state["hombro"]
        p1 = base_robot
        p2 = (p1[0] + L_brazo * math.cos(ang_h), p1[1] - L_brazo * math.sin(ang_h))
        
        # Eslabón 2
        ang_c = math.radians(robot_state["codo"])
        p3 = (p2[0] + L_brazo * math.cos(ang_h - ang_c), p2[1] - L_brazo * math.sin(ang_h - ang_c))

        # Dibujar Robot
        pygame.draw.rect(screen, (40, 50, 80), (base_robot[0]-40, base_robot[1], 80, 40)) # Base
        draw_link(screen, p1, p2, 15, (40, 120, 220))
        draw_link(screen, p2, p3, 15, (40, 120, 220))
        
        # Gripper
        ang_m = math.radians(robot_state["munieca"])
        draw_gripper(screen, p3, ang_m, robot_state["apertura"])

        # HUD / Info
        lado_txt = "DERECHA" if selected_side == "RIGHT" else "IZQUIERDA"
        col_st = (0, 255, 0) if robot_state["detected_pose"] else (200, 50, 50)
        
        screen.blit(font_small.render(f"Configuración: {lado_txt}", True, (200, 200, 200)), (offset_x + 20, 20))
        screen.blit(font_small.render(f"Brazo Detectado", True, col_st), (offset_x + 20, 50))
        
        info_codo = f"Codo: {robot_state['codo']:.1f}°"
        info_grip = f"Apertura: {int(robot_state['apertura']*100)}%"
        screen.blit(font_small.render(info_codo, True, TXT_COLOR), (offset_x + 20, 90))
        screen.blit(font_small.render(info_grip, True, TXT_COLOR), (offset_x + 20, 120))

        # Botón Volver (Instrucción)
        esc_txt = font_small.render("[ESC] Volver al Menú", True, (100, 100, 100))
        screen.blit(esc_txt, (WINDOW_W - 180, WINDOW_H - 30))

    pygame.display.flip()
    clock.tick(30)

cap.release()
pygame.quit()