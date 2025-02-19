import numpy as np 
import matplotlib.pyplot as plt 
import struct
from matplotlib.patches import Wedge, Circle
import json
import base64
import io
from matplotlib.patches import Wedge, Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.io as pio
import plotly.graph_objects as go
import math
import plotly.express as px


class Decoder():
    def __init__(self, data=None):
        # Si no se pasa data, la cargamos desde un archivo
        if data is None:
            # Aquí puedes modificar la ruta según corresponda
            data = read_hex_file("aztechsat-decoder-python-1.0.0/beacon_test.txt")
        self.data = data
    
    def to_int(self, data):
        return int.from_bytes(data, byteorder="big", signed=True)

    def to_uint(self, data):
        return int.from_bytes(data, byteorder="big", signed=False)

    def to_float(self, data):
        return struct.unpack('>f', data)[0]
    
    def get_frame_id(self):
        decoded = json.loads(self.decode_to_json())
        return decoded['clock_from_satellite']

    def _build_data_dict(self):
        frame = self.data
        data = {
            'ax100_hk': {
                'temp_pa': self.to_int(frame[44:46]) / 10.0,
                'last_rssi': self.to_int(frame[46:48]),
                'bgnd_rssi': self.to_int(frame[48:50]),
                'boot_cause': self.to_uint(frame[54:58])
            },
            'pyl_hk': {
                'lm70_temp': self.to_uint(frame[102:104]) / 10.0
            },
            'obc_hk': {
                'temp_a': self.to_int(frame[50:52]) / 10.0,
                'cur_pwm': self.to_uint(frame[52:54]),
                'boot_cause': self.to_uint(frame[54:58]),
                'clock_from_satellite': (frame[104] << 24) + (frame[105] << 16) + (frame[106] << 8) + (frame[108]),
                'magneto_x': self.to_float(frame[58:62]),
                'magneto_y': self.to_float(frame[62:66]),
                'magneto_z': self.to_float(frame[66:70]),
                'gyro_x': self.to_float(frame[70:74]),
                'gyro_y': self.to_float(frame[74:78]),
                'gyro_z': self.to_float(frame[78:82])
            },
            'eps_hk': {
                'vboost': [self.to_uint(frame[0:2]), self.to_uint(frame[2:4]), self.to_uint(frame[4:6])],
                'vbatt': self.to_uint(frame[6:8]),
                'curout': [self.to_uint(frame[18:20]), self.to_uint(frame[20:22]), self.to_uint(frame[22:24]), self.to_uint(frame[24:26])],
                'curin': [self.to_uint(frame[8:10]), self.to_uint(frame[10:12]), self.to_uint(frame[12:14])],
                'cursun': self.to_uint(frame[14:16]),
                'cursys': self.to_uint(frame[16:18]),
                'temp': self.to_int(frame[42:44]),
                'output': [frame[26], frame[27], frame[28], frame[29]],
                'wdt_i2c_time_left': self.to_uint(frame[30:34]),
                'wdt_gnd_time_left': self.to_uint(frame[34:38]),
                'counter_wdt_gnd': self.to_uint(frame[38:42]),
                'boot_cause': self.to_uint(frame[54:58])
            }
        }
        return data

    
    def decode_to_graph(self):
        data = self._build_data_dict()
        ax100_temp_pa, ax100_t_last_rssi, ax100_gnd_rssi = self.ax100_graphs(data)
        pyl_lm70_temp = self.pyl_graphs(data)
        obc_temp_a, obc_cur_pwm, obc_clock, _, _ = self.obc_graphs(data)
        
        return (ax100_temp_pa, ax100_t_last_rssi, ax100_gnd_rssi, 
                pyl_lm70_temp, obc_temp_a, obc_cur_pwm, obc_clock)

    def interactive_graph(self):
        data = self._build_data_dict()
        # Obtenemos los gráficos de OBC y retornamos el gráfico interactivo (asumido como el cuarto elemento)
        _, _, _, obc_magneto, obc_gyro = self.obc_graphs(data)
        p31u_board = self.eps_graphs(data)

        return obc_magneto, obc_gyro, p31u_board
    
    def ax100_graphs(self, data):
        #Crear figuras de AX100 
        ax100_temp_pa = gauge_chart(data['ax100_hk']['temp_pa'], title="Temperature")
        ax100_t_last_rssi = gauge_chart(data['ax100_hk']['last_rssi'], title="Last RSSI")
        fig3, ax3 = plt.subplots()
        ax3.bar("GND RSSI", data['ax100_hk']['bgnd_rssi'], color='blue', width=0.2, edgecolor='black')
        ax3.set_title("GND RSSI")
        for bar in ax3.patches:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', color='white', size=12)

        return ax100_temp_pa, ax100_t_last_rssi, fig3
    
    def pyl_graphs(self, data):
        #Crear figuras de Pylons
        pyl_lm70_temp = gauge_chart(data['pyl_hk']['lm70_temp'], title="LM70 Temperature")
        return pyl_lm70_temp

    def obc_graphs(self, data):
        #Crear figuras de OBC
        obc_temp_a = gauge_chart(data['obc_hk']['temp_a'], title="Temperature Sensor A")

        texto = (f"Current draw for PWM = {data['obc_hk']['cur_pwm']}")
        obc_cur_pwm = plt.figure(figsize=(6,3))
        plt.text(0.5, 0.5, texto, fontsize=15, color='black', ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')

        
        
        obc_boot_cause = gauge_chart(data['obc_hk']['clock_from_satellite'], title="Current clock value")

        texto = (f"Satellite Clock = {data['obc_hk']['clock_from_satellite']}") 
        obc_clock = plt.figure(figsize=(6,3))
        plt.text(0.5, 0.5, texto, fontsize=15, color='black', ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')

        txt = "Magneto X:{:.2f}     Magneto Y:{:.2f}    Magneto Z:{:.2f}".format(data['obc_hk']['magneto_x'],
                                                                                data['obc_hk']['magneto_y'],
                                                                                data['obc_hk']['magneto_z'])
        obc_magneto = plot_3d_oriented_cube(data['obc_hk']['magneto_x'], 
                                            data['obc_hk']['magneto_y'],
                                            data['obc_hk']['magneto_z'], title=txt, clr="red")


        txt = "Gyro X:{:.2f}     Gyro Y:{:.2f}    Gyro Z:{:.2f}".format(data['obc_hk']['gyro_x'],
                                                                        data['obc_hk']['gyro_y'],
                                                                        data['obc_hk']['gyro_z'])        
        obc_gyro = plot_3d_oriented_cube(data['obc_hk']['gyro_x'], 
                                        data['obc_hk']['gyro_y'],
                                        data['obc_hk']['gyro_z'], title=txt)



        return obc_temp_a, obc_cur_pwm, obc_clock, obc_magneto, obc_gyro
    
    def eps_graphs(self, data):
        # general variables
        ambience_tempConst = 25.00 # ambience constant unit = Celsius
        def_temp = float(0) # default temperature
        def_temp = ambience_tempConst - 5 # Hypothetical board overall temperature, according to ambience

        # main constants
        EPS_HK_TEMP_T1 = 18
        EPS_HK_TEMP_T2 = 25
        EPS_HK_TEMP_T3 = 36
        EPS_HK_TEMP_T4 = data['eps_hk']['temp'] # Battery temperature, sensor T4

        # board matrix, initialize default temperatures for all cells
        board_matrix = np.full((27, 26), def_temp, dtype=float)
        # -------------------------------------------------------------
        # make board and show my plot
        board_p31u = makeBoard(EPS_HK_TEMP_T1, EPS_HK_TEMP_T2, EPS_HK_TEMP_T3, EPS_HK_TEMP_T4, board_matrix)
        p31u_heatmap = plot_heatmap(board_p31u)

        return p31u_heatmap

def read_file(ruta):
    file = open(ruta, "rb")
    byte = file.read()
    if b'<HK>' in byte and b'</HK>' in byte:
        inicio = (byte.find(b'<HK>'))
        fin = (byte.find(b'</HK>'))
        return byte[inicio+4:fin]
    else:
        print("Frame Invalido")
        return "NO DATA"
    
def read_file_csv(ruta):
    file = open(ruta, "r")
    byte = file.read()
    for linea_str in byte.split('\n'):
        try:
            date = linea_str.split('|')[0]
            linea = bytes.fromhex(linea_str.split('|')[1])
            if b'<HK>' in linea and b'</HK>' in linea:
                inicio = (linea.find(b'<HK>'))
                fin = (linea.find(b'</HK>'))
                frame = linea[inicio+4:fin]
                return frame
            else:
                print("Frame Invalido")
                return "NO DATA"
        except:
            print("Ocurrio un error")
            return "NO DATA"
        
def read_hex_file(filepath):
    with open(filepath, "r") as file:
        data = file.read().strip()  # Leer y quitar espacios en blanco
    return extract_frame(data)  # Convertir a bytes

def extract_frame(data):
    try:
        linea = bytes.fromhex(data)
        if b'<HK>' in linea and b'</HK>' in linea:
            inicio = (linea.find(b'<HK>'))
            fin = (linea.find(b'</HK>'))
            frame = linea[inicio+4:fin]
            return frame
        else:
            print("Frame Invalido")
            return "NO DATA"
    except:
        print("Ocurrio un error")
        return "NO DATA"
    
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
    
    return fig

def plot_3d_oriented_cube(mag_x, mag_y, mag_z, title="Titulo", clr="blue"):
    size = 1  # Tamaño base del cubo

    # Definir los vértices de un cubo centrado en el origen
    vertices = np.array([[-size, -size, -size],
                         [ size, -size, -size],
                         [ size,  size, -size],
                         [-size,  size, -size],
                         [-size, -size,  size],
                         [ size, -size,  size],
                         [ size,  size,  size],
                         [-size,  size,  size]])

    # Calcular la magnitud original
    magnitude = np.linalg.norm([mag_x, mag_y, mag_z])
    if magnitude == 0:
        magnitude = 1  # Evitar división entre cero

    # Normalización para obtener la dirección
    dir_x, dir_y, dir_z = mag_x / magnitude, mag_y / magnitude, mag_z / magnitude

    # Calcular los ángulos de rotación
    theta_x = np.arctan2(dir_y, dir_z)  # Rotación respecto al eje X
    theta_y = np.arctan2(dir_x, dir_z)  # Rotación respecto al eje Y

    # Matrices de rotación
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])
    
    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])

    # Aplicar rotación
    rotated_vertices = vertices @ R_x.T @ R_y.T

    # Escalar los vértices
    scaled_vertices = rotated_vertices * magnitude

    # Definir las caras correctamente
    i = [0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5, 6, 6, 7]
    j = [1, 3, 4, 2, 5, 3, 6, 7, 5, 7, 0, 6, 1, 7, 2, 3]
    k = [3, 4, 7, 3, 6, 6, 7, 6, 7, 0, 1, 7, 2, 3, 6, 7]

    # Crear figura con Mesh3d
    fig = go.Figure(data=[go.Mesh3d(
        x=scaled_vertices[:, 0],
        y=scaled_vertices[:, 1],
        z=scaled_vertices[:, 2],
        i=i, j=j, k=k,
        color=clr,
        opacity=1
    )])


    # Ajustar los límites de los ejes dinámicamente
    axis_range = magnitude * 1.5
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-axis_range, axis_range]),
            yaxis=dict(range=[-axis_range, axis_range]),
            zaxis=dict(range=[-axis_range, axis_range]),
            aspectmode="cube"
        ),
        title=f"{title}",
        margin=dict(r=0, l=0, b=0, t=40)
    )

    return fig

        #           for HEATMAP
    # ______---GENERATE BOARD---______
def makeBoard(T1, T2, T3, T4, board_matrix):
    # define the sensor temperatures
    board_matrix[23][4] = T1  # T1
    board_matrix[0][7] = T2  # T2
    board_matrix[23][21] = T3  # T3
    board_matrix[13][23] = T4  # T4

    # fill matrix with values using idw
    for i in range(27):
        for j in range(26):
            if (not isASensor(i, j)):
                board_matrix[i][j] = idw(i, j, board_matrix)
    
    return board_matrix
    

# _____--- MATPLOTLIB ---______
# matplot enters the chat
def plot_heatmap(board_matrix):
    fig = px.imshow(
        board_matrix,
        labels=dict(x="Eje X", y="Eje Y", color="Temperatura"),
        x=[f"Columna {i}" for i in range(board_matrix.shape[1])],
        y=[f"Fila {i}" for i in range(board_matrix.shape[0])],
        color_continuous_scale='plasma',
        origin='lower',
        text_auto=True
    )
    fig.update_layout(title="Mapa de Calor de la Placa")
    return fig



# is the coordinate within the sensor list? (boolean function)
def isASensor(i, j):
    if (i == 23 and j == 4) or (i == 0 and j == 7) or (i == 23 and j == 21) or (i == 13 and j == 23):
        return True
    else:
        return False

# Euclidean distance function
def euDist(array1, array2):
    xdiff = array2[0] - array1[0]
    ydiff = array2[1] - array1[1]
    return math.sqrt(abs((pow(xdiff, 2)) + (pow(ydiff, 2))))


# ______---IDW---______
# the Inverse Distance Weighted Interpolation formula,
# What it does? ... returns an approximate float value of the temperature relative to known points

# Inverse Distance function
def weight_i(arr):
    beta = 1 # measures how important the distance is. Values = 0, 1 or 2
    return (1/(pow(arr[3], beta)))

# Product of Inverse Distance and the respective Value. Funciton
def wi_zi(arr):
    return (arr[4] * arr[2]) # weight times value

# Whole IDW function
def idw(i, j, board_matrix):
    # constants
    weight_sum = 0.0 # weight
    weight_z_sum = 0.0 # wi times zi (z is the value of the current cell)
    idw_res = 0.0 # result 

    # sensor data basis:
    # (known points)
    #              x,  y,      value,  dist(u, i) wi wizi
    t1 = np.array([4, 23, board_matrix[23][4], 0, 0, 0], dtype=float)
    t2 = np.array([7, 0, board_matrix[0][7], 0, 0, 0], dtype=float)
    t3 = np.array([21, 23, board_matrix[23][21], 0, 0, 0], dtype=float)
    t4 = np.array([23, 13, board_matrix[13][23], 0, 0, 0], dtype=float)

    # unknown point (given, to determine)
    upoint = np.array([j, i, 0, 0, 0, 0], dtype=float)
    
    masterArray = np.array([t1, t2, t3, t4]) # array to fill the data from the temp. sensors
    
    for i in masterArray:
        i[3] = euDist(i, upoint) # saves the distance(CurrentPoint to  UnknownPoint)
        i[4] = weight_i(i) # saves the inverse distance value (weight)
        i[5] = wi_zi(i) # saves the product of the i-point value and the inverse distance (weight)
        # for overall sums
        weight_sum += i[4]  
        weight_z_sum += i[5]

    idw_res = weight_z_sum/weight_sum
    return idw_res


def fig_to_uri(fig, anim=None, fmt=None):
    """
    Convierte una figura o animación de Matplotlib en un URI base64 compatible con Dash.
    
    - `fig`: Figura de Matplotlib (para gráficos estáticos) o Plotly (para gráficos interactivos).
    - `anim`: Animación de Matplotlib (para gráficos animados).
    - `fmt`: Formato deseado (puede ser "png", "svg", "gif", "mp4"). Si es None, se detecta automáticamente.
    """
    buf = io.BytesIO()

    # Si es una animación
    if anim:
        fmt = fmt or "gif"  # Predeterminado a GIF para animaciones
        if fmt == "gif":
            anim.save(buf, format="gif", writer="pillow")
        elif fmt == "mp4":
            anim.save(buf, writer="ffmpeg", fps=10)
        else:
            raise ValueError("Formato no compatible para animaciones. Usa 'gif' o 'mp4'.")
    
    # Si es un gráfico estático (Matplotlib o Plotly)
    else:
        fmt = fmt or "png"  # Predeterminado a PNG para gráficos estáticos
        if fmt not in ["png", "svg"]:
            raise ValueError("Formato no compatible para imágenes estáticas. Usa 'png' o 'svg'.")

        # Si la figura es de Plotly
        if isinstance(fig, go.Figure):  # Si es una figura de Plotly
            pio.write_image(fig, buf, format=fmt)  # Usamos write_image de Plotly para generar la imagen
        else:
            fig.savefig(buf, format=fmt)  # Si es una figura de Matplotlib

    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    
    if fmt in ["png", "svg", "gif"]:
        return f"data:image/{fmt};base64,{img_str}"
    elif fmt == "mp4":
        return f"data:video/mp4;base64,{img_str}"
    else:
        raise ValueError("Formato no soportado.")