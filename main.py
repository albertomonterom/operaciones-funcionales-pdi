import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Variables globales para las imagenes
img1 = None
img2 = None
gris = None

def mostrar_imagen(img):
    # Convertir la imagen de BGR a RGB para mostrarla correctamente
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    lbl_imagen.config(image=img_tk)
    lbl_imagen.image = img_tk

# Cargamos la primera imagen
def cargar_imagen1():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg;*.bmp")])
    if filepath:
        global img1
        img1 = cv2.resize(cv2.imread(filepath), (256, 256))
        mostrar_imagen(img1)

# Cargamos la segunda imagen
def cargar_imagen2():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg;*.bmp")])
    if filepath:
        global img2
        img2 = cv2.resize(cv2.imread(filepath), (256, 256))
        mostrar_imagen(img2)

# Operaciones aritmeticas
def operacion_aritmetica(operacion):
    if img1 is None or img2 is None:
        return messagebox.showerror("Error", "Debe seleccionar dos imágenes antes de realizar alguna operación")
    if img1 is not None and img2 is not None:
        if operacion == 'Suma':
            resultado = cv2.add(img1, img2)
        elif operacion == 'Resta':
            resultado = cv2.subtract(img1, img2)
        elif operacion == 'Multiplicacion':
            resultado = cv2.multiply(img1, img2)
        mostrar_imagen(resultado)

# Operaciones aritmeticas con escalar
def operacion_aritmetica_escalar(operacion):
    if img1 is None:
        return messagebox.showerror("Error", "Debe seleccionar una imagen antes de realizar la operación")
    if img1 is not None:
        valor_escalar = valida_escalar()
        if operacion == 'Sumar Escalar':
            resultado = cv2.add(img1, valor_escalar)
        elif operacion == 'Restar Escalar':
            resultado = cv2.subtract(img1, valor_escalar)
        elif operacion == 'Multiplicar Escalar':
            resultado = cv2.multiply(img1, valor_escalar)
        mostrar_imagen(resultado)

# Operaciones lógicas
def operacion_logica(operacion):
    if img1 is None or img2 is None:
        return messagebox.showerror("Error", "Debe seleccionar dos imágenes antes de realizar alguna operación")
    if img1 is not None and img2 is not None:
        if operacion == 'AND':
            resultado = cv2.bitwise_and(img1, img2)
        elif operacion == 'OR':
            resultado = cv2.bitwise_or(img1, img2)
        elif operacion == 'XOR':
            resultado = cv2.bitwise_xor(img1, img2)
        mostrar_imagen(resultado)

# Convertir a escala de grises
def convertir_escala_grises():
    if img1 is None:
        return messagebox.showerror("Error", "Debe seleccionar una imagen")
    if img1 is not None:
        gris = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        mostrar_imagen(gris)

# Umbralizado
def umbralizar_imagen():
    if img1 is None:
        return messagebox.showerror("Error", "Debe seleccionar una imagen")
    if img1 is not None:
        umbral = valida_umbral()
        gris = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        _, img_umbralizada = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)
        mostrar_imagen(img_umbralizada)

# Mostrar el histograma
def mostrar_histograma():
    if img1 is None:
        return messagebox.showerror("Error", "Debe seleccionar una imagen")
    if img1 is not None:
        gris = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        plt.hist(gris.ravel(), 256, [0, 256])
        plt.show()

# Componentes espectrales
def extraer_componentes_espectrales():
    if img1 is None:
        return messagebox.showerror("Error", "Debe seleccionar una imagen")
    if img1 is not None:
        # Separar los componentes de color
        b, g, r = cv2.split(img1)

        # Crear imagenes individuales para cada canal
        b_img = np.zeros_like(img1)
        g_img = np.zeros_like(img1)
        r_img = np.zeros_like(img1)

        # Asignar los valores correspondientes a cada canal
        b_img[:, :, 0] = b # Canal azul
        g_img[:, :, 1] = g # Canal verde
        r_img[:, :, 2] = r # Canal rojo

        # Convertir las imagenes en formato adecuado para mostrarlas en Tkinter
        b_img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)))
        g_img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB)))
        r_img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)))

        lbl_imagen.destroy()

        # Mostrar las imagenes en las etiquetas (labels)
        lbl_blue.config(image=b_img_tk)
        lbl_blue.image = b_img_tk

        lbl_green.config(image=g_img_tk)
        lbl_green.image = g_img_tk

        lbl_red.config(image=r_img_tk)
        lbl_red.image = r_img_tk

# Corrección gamma
def correccion_gamma():
    if img1 is None:
        return messagebox.showerror("Error", "Debe seleccionar una imagen")
    if img1 is not None:
        gamma = valida_gamma()
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            img_corregida = cv2.LUT(img1, lookUpTable)
            mostrar_imagen(img_corregida)

# Ecualización exponencial
def ecualizacion_exponencial():
    if img1 is None:
        return messagebox.showerror("Error", "Debe seleccionar una imagen")
    # Convertir a escala de grises
    imagen = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # Normalizar la imagen a rango [0, 1]
    imagen_normalizada = imagen / 255.0
    # Calcular el histograma y la función de probabilidad acumulada (CDF)
    histograma, _ = np.histogram(imagen, bins=256, range=(0, 255))
    cdf = np.cumsum(histograma) / np.sum(histograma)
    # Parámetros para la ecualización exponencial
    alpha = 1
    g_min = 0
    # Evitar problemas de logaritmos
    cdf = np.clip(cdf, 0, 1 - 1e-6)  # Asegurar que CDF no sea 1
    # Aplicar la fórmula de ecualización exponencial a cada píxel
    imagen_exponencial = g_min - (1 / alpha) * np.log(1 - cdf[imagen.astype(int)])  # Convertir a int para el índice
    # Normalizar los valores de vuelta al rango [0, 255] y convertir a uint8
    imagen_exponencial = np.clip(imagen_exponencial * 255, 0, 255).astype(np.uint8)
    # Mostrar imagen
    mostrar_imagen(imagen_exponencial)



# Validar escalar
def valida_escalar():
    # Valida que el valor del escalar sea numérico y no esté vació
    valor_escalar = entry_escalar.get()
    if not valor_escalar: # Verifica si el campo está vacío
        return messagebox.showerror("Error", "Debes ingresar un valor escalar")
    else:
        return int(entry_escalar.get())

def valida_umbral():
    # Valida que el valor del umbral sea numérico y no esté vacío
    umbral = entry_umbral.get()
    if not umbral: # Verifica si el campo está vacío
        return messagebox.showerror("Error", "Debes ingresar un valor umbral")
    else:
        return int(entry_umbral.get())

def valida_gamma():
    # Valida que el valor de Gamma sea numérico y no esté vacío
    gamma = entry_gamma.get()
    if not gamma: # Verifica si el campo está vacío
        return messagebox.showerror("Error", "Debes ingresar un valor")
    else:
        return float(entry_gamma.get())


# Definimos la interfaz haciendo uso de la librería tkinter incluida ya en Python y sus dimensiones
root = tk.Tk()
root.title("Procesamiento de Imagenes")
root.geometry("900x1000")

# Cargar imagen 1
btn_cargar1 = tk.Button(root, text="Cargar imagen 1", command=cargar_imagen1)
btn_cargar1.pack(padx=5)

# Cargar imagen 2
btn_cargar2 = tk.Button(root, text="Cargar Imagen 2", command=cargar_imagen2)
btn_cargar2.pack()

# Mostrar imagen
lbl_imagen = tk.Label(root)
lbl_imagen.pack()

frame_colores = tk.Frame(root)
frame_colores.pack(pady=5)

lbl_blue = tk.Label(frame_colores)
lbl_blue.pack(side=tk.LEFT, padx=10)

lbl_green = tk.Label(frame_colores)
lbl_green.pack(side=tk.LEFT, padx=10)

lbl_red = tk.Label(frame_colores)
lbl_red.pack(side=tk.LEFT, padx=10)

# Entrada para el valor escalar
frame_escalar = tk.Frame(root)
frame_escalar.pack(pady=5)
tk.Label(frame_escalar, text="Valor Escalar").pack(side=tk.LEFT)
entry_escalar = tk.Entry(frame_escalar)
entry_escalar.pack(side=tk.LEFT, padx=5)

# Operaciones aritméticas con escalar
frame_operaciones_escalar = tk.Frame(root)
frame_operaciones_escalar.pack(pady=5)
tk.Label(frame_operaciones_escalar, text="Operaciones Aritméticas con Escalar").pack(side=tk.LEFT)
operaciones_aritmeticas_escalar = ttk.Combobox(frame_operaciones_escalar, values=["Sumar Escalar", "Restar Escalar", "Multiplicar Escalar"])
operaciones_aritmeticas_escalar.pack(side=tk.LEFT, padx=5)
btn_operacion_aritmetica_escalar = tk.Button(frame_operaciones_escalar, text="Aplicar", command=lambda: operacion_aritmetica_escalar(operaciones_aritmeticas_escalar.get()))
btn_operacion_aritmetica_escalar.pack(side=tk.LEFT, padx=5)

# Operaciones aritméticas
frame_operaciones = tk.Frame(root)
frame_operaciones.pack(pady=5)
tk.Label(frame_operaciones, text="Operaciones Aritméticas").pack(side=tk.LEFT)
operaciones_aritmeticas = ttk.Combobox(frame_operaciones, values=["Suma", "Resta", "Multiplicacion"])
operaciones_aritmeticas.pack(side=tk.LEFT, padx=5)
btn_operacion_aritmetica = tk.Button(frame_operaciones, text="Aplicar", command=lambda: operacion_aritmetica(operaciones_aritmeticas.get()))
btn_operacion_aritmetica.pack(side=tk.LEFT, padx=5)

# Operaciones lógicas
frame_logicas = tk.Frame(root)
frame_logicas.pack(pady=5)
tk.Label(frame_logicas, text="Operaciones Lógicas").pack(side=tk.LEFT)
operaciones_logicas = ttk.Combobox(frame_logicas, values=["AND", "OR", "XOR"])
operaciones_logicas.pack(side=tk.LEFT, padx=5)
btn_operacion_logica = tk.Button(frame_logicas, text="Aplicar", command=lambda: operacion_logica(operaciones_logicas.get()))
btn_operacion_logica.pack(side=tk.LEFT, padx=5)

# Umbralizar imagen
frame_umbral = tk.Frame(root)
frame_umbral.pack(pady=5)
tk.Label(frame_umbral, text="Umbral").pack(side=tk.LEFT)
entry_umbral = tk.Entry(frame_umbral)
entry_umbral.pack(side=tk.LEFT, padx=5)
btn_umbralizar = tk.Button(frame_umbral, text="Umbralizar", command=umbralizar_imagen)
btn_umbralizar.pack(side=tk.LEFT, padx=5)

# Convertir a escala de grises
btn_grises = tk.Button(root, text="Convertir a Escala de Grises", command=convertir_escala_grises)
btn_grises.pack()


# Mostrar histograma
btn_histograma = tk.Button(root, text="Mostrar Histograma", command=mostrar_histograma)
btn_histograma.pack()

# Extraer componentes espectrales
btn_componentes = tk.Button(root, text="Extraer Componentes Espectrales", command=extraer_componentes_espectrales)
btn_componentes.pack()

# Ecualización exponencial
btn_ecualizacion_exponencial = tk.Button(root, text="Ecualización Exponencial", command=ecualizacion_exponencial)
btn_ecualizacion_exponencial.pack()

# Aplicar corrección Gamma
frame_gamma = tk.Frame(root)
frame_gamma.pack(pady=5)
tk.Label(frame_gamma, text="Corrección Gamma").pack(side=tk.LEFT)
entry_gamma = tk.Entry(frame_gamma)
entry_gamma.pack(side=tk.LEFT, padx=5)
btn_gamma = tk.Button(frame_gamma, text="Aplicar Corrección Gamma", command=correccion_gamma)
btn_gamma.pack(side=tk.LEFT, padx=5)

root.mainloop()