import numpy as np
from ezllm.core import Model
from ezllm.layers import Dense, ReLU
from ezllm.losses import cross_entropy_loss
from ezllm.tokenizer import Tokenizer

def test_talk():
    # Conjunto de dados trivial de conversa:
    # Mapeamentos de palavra de entrada para palavra de saída.
    inputs = ["oi", "olá", "tudo", "como", "vai"]
    outputs = ["olá", "oi", "bem", "vai", "bem"]

    # O tokenizador é treinado com todas as palavras envolvidas.
    all_texts = inputs + outputs
    tokenizer = Tokenizer(lower=True, sep=" ")
    tokenizer.fit(all_texts)
    
    # Utiliza a codificação one-hot automática do tokenizador
    X = tokenizer.encode_onehot(inputs)
    y = tokenizer.encode_onehot(outputs)
    
    # Define um modelo simples: camada Dense -> ReLU -> camada Dense
    model = Model([
        Dense(X.shape[1], 10),
        ReLU(),
        Dense(10, X.shape[1])
    ])
    
    # Treina o modelo utilizando a cross entropy loss
    print("Treinando a rede neural para 'aprender a falar'...")
    model.fit(X, y, epochs=1000, lr=0.1, loss_fn=cross_entropy_loss, verbose=0)
    
    # Testa o modelo em cada exemplo de entrada
    print("Resultados do teste:")
    for i, word in enumerate(inputs):
        x = X[i:i+1]
        # Utiliza o método predict do tokenizador para obter a predição diretamente
        predicted_word = tokenizer.predict(model.predict(x))
        print(f"Entrada: {word} -> Predição: {predicted_word}, Alvo: {outputs[i]}")
        # Verifica se a predição coincide com o alvo
        assert predicted_word == outputs[i], f"O modelo errou a predição para '{word}' (esperado: '{outputs[i]}', obtido: '{predicted_word}')"

if __name__ == "__main__":
    test_talk()
    print("Teste de rede neural que aprende a falar passou!") 