import numpy as np
from ezllm.core import Model
from ezllm.tokenizer import Tokenizer
from ezllm.losses import cross_entropy_loss
from ezllm.layers import Dense, ReLU
from PIL import Image

# Carrega as imagens
image = Image.open("image.png")
image2 = Image.open("image2.png")

# Cria e ajusta o tokenizador com ambas as imagens
tokenizer = Tokenizer(lower=False, sep=None)

# Ajusta o tokenizador com a primeira imagem
tokenizer.fit_image(image)

# Adiciona as cores da segunda imagem sem sobrescrever
tokenizer.fit_image_again(image2)

# Codifica as imagens
X = tokenizer.encode_image_onehot(image)
y = X
X2 = tokenizer.encode_image_onehot(image2)
y2 = X2

# Cria o modelo com dimensões fixas
dim = len(tokenizer.word2idx)  # Usa o tamanho total do vocabulário
test_model = Model([
    Dense(dim, 64),
    ReLU(),
    Dense(64, 128),
    ReLU(),
    Dense(128, 128),
    ReLU(),
    Dense(128, dim)
])

print("Treinando primeira imagem...")

# Primeiro treinamento
test_model.fit(X, y,
             epochs=500,
             lr=0.5,
             loss_fn=cross_entropy_loss,
             verbose=0,
             progress_bar=True,
             output_path="ai.png",
             img_size=image.size,
             output_test_train=True,
             output_interval=20,
             test_train=True,
             tokenizer=tokenizer
        )

print("Treinando segunda imagem...")

# Segundo treinamento
test_model.fit_another_image([image2], [image2],
             epochs=200,
             lr=0.5,
             loss_fn=cross_entropy_loss,
             verbose=0,
             progress_bar=True,
             output_path="ai.png",
             img_size=image2.size,
             output_test_train=True,
             output_interval=20,
             test_train=True,
             tokenizer=tokenizer
        )
    
list_images = [image, image2]
list_target_images = [image, image2]

test_model.adjust_model_toBetter_inImage(list_images, list_target_images,
                                        tokenizer=tokenizer,
                                        epochs=200,
                                        lr=0.5,
                                        loss_fn=cross_entropy_loss,
                                        img_size=(image.size, image2.size))

print("Treinamento concluído!")

print("Predizendo...")

# Predição e salvamento da imagem
pred = test_model.predict(X)
tokenizer.save_image(pred, "ai.png", original_size=image.size)

print("Predição concluída!")

# Salva o modelo
test_model.save("test_model.pkl")