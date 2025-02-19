import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

def gauge_chart(value, min_val=0, max_val=100, title="Gauge Chart"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.6, 1.2)

    sections = [
        (120, 180, '#4CAF50'),   # Verde
        (60, 120, '#FFC107'),    # Amarillo
        (0, 60, '#FF3B3B')      # Rojo
    ]
    
    for start, end, color in sections:
        wedge = Wedge((0, 0), 1, start, end, facecolor=color, edgecolor='black', lw=1.5)
        ax.add_patch(wedge)

    angle = 180 * (1 - (value - min_val) / (max_val - min_val))
    needle_x = [0, 0.8 * np.cos(np.radians(angle))]
    needle_y = [0, 0.8 * np.sin(np.radians(angle))]
    ax.plot(needle_x, needle_y, color='black', linewidth=3, solid_capstyle='round')

    center_circle = Circle((0, 0), 0.05, color='black', zorder=10)
    ax.add_patch(center_circle)

    for i in range(0, 101, 20):
        angle = 180 * (1 - (i - min_val) / (max_val - min_val))
        x = 1.05 * np.cos(np.radians(angle))
        y = 1.05 * np.sin(np.radians(angle))
        ax.text(x, y, str(i), ha='center', va='center', fontsize=10, fontweight='bold')

    ax.text(0, -0.35, f'{value}%', ha='center', fontsize=14, fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def plot_3d_oriented_cube(mag_x, mag_y, mag_z, title = "Titulo"):
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    size = 1  # Tamaño del cubo
    vertices = np.array([[-size, -size, -size],
                         [ size, -size, -size],
                         [ size,  size, -size],
                         [-size,  size, -size],
                         [-size, -size,  size],
                         [ size, -size,  size],
                         [ size,  size,  size],
                         [-size,  size,  size]])

    norm = np.linalg.norm([mag_x, mag_y, mag_z])
    if norm == 0:
        norm = 1
    mag_x, mag_y, mag_z = mag_x / norm, mag_y / norm, mag_z / norm

    theta_x = np.arctan2(mag_y, mag_z)  # Rotación respecto al eje X
    theta_y = np.arctan2(mag_x, mag_z)  # Rotación respecto al eje Y

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])
    
    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])

    rotated_vertices = vertices @ R_x.T @ R_y.T

    faces = [[rotated_vertices[j] for j in [0, 1, 2, 3]],
             [rotated_vertices[j] for j in [4, 5, 6, 7]],
             [rotated_vertices[j] for j in [0, 1, 5, 4]],
             [rotated_vertices[j] for j in [2, 3, 7, 6]],
             [rotated_vertices[j] for j in [1, 2, 6, 5]],
             [rotated_vertices[j] for j in [0, 3, 7, 4]]]

    ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='black', alpha=0.5))

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Eje Z')

    # Mostrar la figura
    plt.title(f"Orientación del {title}")



#Variables PYL
AGUILA_BOARD_TEMP = 100

#Variables OBC
A3200_TEMP_A = 1
A3200_TEMP_B = 1
A3200_CUR_GSSB1 = 1
A3200_CUR_GSSB2 = 1
A3200_CUR_PWM = 1
A3200_BOOT_COUNTER = 1
A3200_BOOT_CAUSE = 1
A3200_CLOCK = 1
A3200_TICKS = 1
MAGEJEX = 1
MAGEJEY  = 100
MAGEJEZ = 2
GIROEJEX = 1
GIROEJEY = 1
GIROEJEZ = 1
SWITCH_PWM = 1
PWM_CURRENT = 1

#Variables EPS
EPS_HK_VBOOST = 1
EPS_HK_VBATT = 1
EPS_HK_CUROUT = 1
EPS_HK_CURIN = 1
EPS_HK_CURSUN = 1
EPS_HK_CURSYS = 1
EPS_HK_TEMP = 1
EPS_HK_OUT_VAL = 1
EPS_HK_WDT_I2C_S = 1
EPS_HK_WDT_GND_S = 1
EPS_HK_CNT_BOOT = 1
EPS_HK_CNT_WDT_GND = 1
EPS_HK_BOOT_CAUSE = 1
EPS_HK_BATTMODE = 1

#Variables AX100
AX100_TELEM_TEMP_BRD = 1
AX100_TELEM_TEMP_PA = 1
AX100_TELEM_LAST_RSSI = 1
AX100_TELEM_LAST_RFERR = 1
AX100_TELEM_BOOT_COUNT = 1
AX100_TELEM_BOOT_CAUSE = 1
AX100_TELEM_BGND_RSSI = 1


#Housekeeping PYL
gauge_chart(AGUILA_BOARD_TEMP, title="Temperatura PYL")

#Housekeeping OBC
gauge_chart(A3200_TEMP_A, title="Temperatura A3200 sensor A")
gauge_chart(A3200_TEMP_B, title="Temperatura A3200 sensor B")

plt.figure(figsize=(6, 3))
plt.text(0, 1, f"Valores Variables OBC", fontsize=30, color='black', ha="left")
texto = (
    f"A3200_CUR_GSSB1 = {A3200_CUR_GSSB1}\n"
    f"A3200_CUR_GSSB2 = {A3200_CUR_GSSB2}\n"
    f"A3200_CUR_PWM = {A3200_CUR_PWM}\n"
    f"A3200_BOOT_COUNTER = {A3200_BOOT_COUNTER}\n"
    f"A3200_CLOCK = {A3200_CLOCK}\n"
    f"A3200_TICKS = {A3200_TICKS}\n"
    f"Switch PWM = {SWITCH_PWM}\n"
    f"PWM current = {PWM_CURRENT}"
)
plt.text(0, 0, texto, fontsize=15, color='black', ha="left")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')

plot_3d_oriented_cube(MAGEJEX, MAGEJEY, MAGEJEZ, title = "Magnetometro A3200")
plot_3d_oriented_cube(GIROEJEX, GIROEJEY, GIROEJEZ, title = "Giroscopio A3200")

opciones = {
    1: "Reset",
    2: "Inicialización",
    3: "Arranque",
    4: "Modo de espera",
    5: "Error",
}

plt.figure(figsize=(5, 3))
plt.text(0,1, "Causa boot OBC A3200", fontsize=12, color='black')
for i, (clave,texto) in enumerate(opciones.items()):
    color = 'green' if clave == A3200_BOOT_CAUSE else 'black' 
    plt.text(0, 0.8 - 0.2 * i, f"{clave} : {texto}", fontsize=12, color=color)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')


#Housekeeping EPS


#Housekeeping AX100

plt.show()