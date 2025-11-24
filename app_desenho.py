import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os

# ==========================================
# PARTE 1: O CÉREBRO DA IA (BACKEND)
# ==========================================

def treinar_modelo_cnn():
    print("--- Iniciando Treinamento da IA (Aguarde...) ---")
    
    # 1. Carregar Dados MNIST
    # O dataset já vem separado em treino e teste
    (x_treino, y_treino), (x_teste, y_teste) = keras.datasets.mnist.load_data()
    
    # 2. Pré-processamento
    # Normalizar pixels de 0-255 para 0-1
    x_treino = x_treino.astype("float32") / 255.0
    # Adicionar dimensão de canal (28, 28, 1) para a CNN
    x_treino = np.expand_dims(x_treino, -1)
    
    # 3. Construir a Arquitetura da Rede Neural (CNN)
    modelo = models.Sequential([
        keras.Input(shape=(28, 28, 1)), # Entrada: Imagem 28x28px em escala de cinza
        
        # Camadas de Convolução (Extraem características como bordas e curvas)
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Achatamento (Prepara para a classificação final)
        layers.Flatten(),
        
        # Dropout (Desliga neurônios aleatoriamente para evitar "decoreba")
        layers.Dropout(0.5),
        
        # Saída (10 neurônios = 10 dígitos possíveis)
        layers.Dense(10, activation="softmax"),
    ])
    
    # 4. Compilar e Treinar
    modelo.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    # Treino rápido (3 épocas são suficientes para demonstração)
    modelo.fit(x_treino, y_treino, batch_size=128, epochs=3, verbose=1)
    print("--- Treinamento Concluído com Sucesso ---")
    
    return modelo

# ==========================================
# PARTE 2: A INTERFACE GRÁFICA (FRONTEND)
# ==========================================

class AppReconhecimento:
    def __init__(self, janela_raiz, modelo):
        self.janela_raiz = janela_raiz
        self.modelo = modelo
        self.janela_raiz.title("Reconhecimento de Dígitos com IA")
        
        # Configurações do Pincel e Tela
        # IMPORTANTE: Fundo PRETO e Pincel BRANCO para igualar ao dataset MNIST
        self.largura_canvas = 280
        self.altura_canvas = 280
        self.cor_fundo = "black"
        self.cor_pincel = "white"
        self.tamanho_pincel = 10
        
        # Elementos da Interface (Layout)
        self.texto_resultado = tk.Label(janela_raiz, text="Desenhe um número...", font=("Helvetica", 18))
        self.texto_resultado.pack(pady=10)
        
        self.canvas = tk.Canvas(janela_raiz, width=self.largura_canvas, height=self.altura_canvas, bg=self.cor_fundo)
        self.canvas.pack()
        
        # Botões
        frame_botoes = tk.Frame(janela_raiz)
        frame_botoes.pack(pady=10)
        
        self.btn_limpar = tk.Button(frame_botoes, text="Limpar Tela", command=self.limpar_tela)
        self.btn_limpar.pack(side=tk.LEFT, padx=10)
        
        self.btn_prever = tk.Button(frame_botoes, text="Identificar", command=self.prever_digito, bg="green", fg="white")
        self.btn_prever.pack(side=tk.LEFT, padx=10)
        
        # Estado do Desenho (Memória)
        # Cria uma imagem vazia na memória para a IA ler depois
        self.imagem_memoria = Image.new("L", (self.largura_canvas, self.altura_canvas), self.cor_fundo)
        self.desenhista = ImageDraw.Draw(self.imagem_memoria)
        
        # Evento: Quando arrastar o mouse, chama a função desenhar
        self.canvas.bind("<B1-Motion>", self.desenhar)

    def desenhar(self, evento):
        # Desenha na tela (para o usuário ver)
        r = self.tamanho_pincel
        x1, y1 = (evento.x - r), (evento.y - r)
        x2, y2 = (evento.x + r), (evento.y + r)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.cor_pincel, outline=self.cor_pincel)
        
        # Desenha na memória (para a IA processar)
        self.desenhista.ellipse([x1, y1, x2, y2], fill=self.cor_pincel, outline=self.cor_pincel)

    def limpar_tela(self):
        self.canvas.delete("all")
        self.imagem_memoria = Image.new("L", (self.largura_canvas, self.altura_canvas), self.cor_fundo)
        self.desenhista = ImageDraw.Draw(self.imagem_memoria)
        self.texto_resultado.config(text="Desenhe um número...")

    def prever_digito(self):
        # 1. Redimensionar a imagem desenhada para 28x28 (Padrão MNIST)
        img_redimensionada = self.imagem_memoria.resize((28, 28))
        
        # 2. Converter para array numérico (matriz)
        array_img = np.array(img_redimensionada)
        
        # 3. Normalizar (converter valores de 0-255 para 0-1)
        array_img = array_img.astype("float32") / 255.0
        
        # 4. Ajustar formato para o modelo (Batch, Altura, Largura, Canais)
        array_img = np.expand_dims(array_img, axis=0) # Adiciona dimensão do lote
        array_img = np.expand_dims(array_img, axis=-1) # Adiciona canal de cor
        
        # 5. Fazer a Predição
        predicao = self.modelo.predict(array_img)
        digito_provavel = np.argmax(predicao)
        confianca = np.max(predicao)
        
        self.texto_resultado.config(text=f"Eu acho que é: {digito_provavel} ({confianca*100:.1f}%)")

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # 1. Treinar o modelo antes de abrir a janela
    modelo_treinado = treinar_modelo_cnn()
    
    # 2. Iniciar a Interface Gráfica
    raiz = tk.Tk()
    app = AppReconhecimento(raiz, modelo_treinado)
    raiz.mainloop()