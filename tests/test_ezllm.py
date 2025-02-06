import numpy as np
from ezllm.core import Model
from ezllm.layers import Dense, ReLU
from ezllm.losses import mse_loss
from ezllm.tokenizer import Tokenizer

# Teste 1: Regressão simples
def test_regression():
    # Gerar dataset simples: y = 2 * x + 1
    X = np.linspace(0, 1, 100).reshape(-1, 1)
    y = 2 * X + 1

    # Modelo: camada densa -> ReLU -> camada densa
    model = Model([
        Dense(1, 10),
        ReLU(),
        Dense(10, 1)
    ])

    print("Treinando o modelo de regressão...")
    model.fit(X, y, epochs=500, lr=0.1, loss_fn=mse_loss, verbose=0)

    y_pred = model.predict(X)
    loss, _ = mse_loss(y_pred, y)
    print("MSE final na regressão:", loss)
    # Verifica se o loss está abaixo de um determinado limiar
    assert loss < 0.1, "O modelo de regressão não convergiu de forma satisfatória."

# Teste 2: Tokenizador simples
def test_tokenizer():
    texts = ["Hello world", "Hello test", "World of Python"]
    tokenizer = Tokenizer(lower=True, sep=" ")
    tokenizer.fit(texts)

    encoded = tokenizer.encode("Hello world")
    decoded = tokenizer.decode(encoded)
    print("Texto codificado:", encoded)
    print("Texto decodificado:", decoded)
    # Verifica se as palavras "hello" e "world" estão na decodificação (em lowercase)
    assert "hello" in decoded and "world" in decoded, "Tokenizador falhou no encode/decode."

if __name__ == "__main__":
    test_regression()
    test_tokenizer()
    print("Todos os testes passaram!") 