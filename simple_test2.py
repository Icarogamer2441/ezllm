from ezllm.core import Model
from ezllm.layers import Dense, ReLU
from ezllm.tokenizer import Tokenizer
from PIL import Image
import numpy as np

# Carrega a imagem
image = Image.open("image.png")
image2 = Image.open("image2.png")

# Cria o tokenizador
tokenizer = Tokenizer(lower=False, sep=None)
tokenizer.fit_image(image)

tokenizer.fit_image_again(image2)

# Cria o modelo
model = Model.load("test_model.pkl")

X = tokenizer.encode_image_onehot(image)
X2 = tokenizer.encode_image_onehot(image2)

# Prediz a imagem
pred = model.predict(X, temperature=0.7)

# Salva a imagem
tokenizer.save_image(pred, "ai.png", original_size=image.size)