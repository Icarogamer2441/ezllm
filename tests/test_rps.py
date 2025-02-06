import numpy as np
import os
from ezllm.core import Model
from ezllm.layers import Dense, ReLU
from ezllm.losses import cross_entropy_loss
from ezllm.tokenizer import Tokenizer

def test_rock_paper_scissors_model():
    # Define os mapeamentos: entrada e a jogada vencedora
    inputs = ["pedra", "papel", "tesoura"]
    outputs = ["papel", "tesoura", "pedra"]

    # Treina o tokenizador com todas as palavras envolvidas
    all_texts = inputs + outputs
    tokenizer = Tokenizer(lower=True, sep=" ")
    tokenizer.fit(all_texts)

    # Utiliza a codificação one-hot automática do tokenizador
    X = tokenizer.encode_onehot(inputs)
    y = tokenizer.encode_onehot(outputs)
    
    # Cria um modelo simples: camada Dense -> ReLU -> camada Dense
    model = Model([
        Dense(X.shape[1], 10),
        ReLU(),
        Dense(10, X.shape[1])
    ])
    
    # Treina o modelo utilizando cross entropy loss com barra de progresso
    print("Treinando o modelo de Pedra, Papel, Tesoura...")
    model.fit(X, y, epochs=1000, lr=0.1, loss_fn=cross_entropy_loss, verbose=0, progress_bar=True)
    
    # Salva o modelo em um arquivo
    save_path = "rps_model.pkl"
    model.save(save_path)
    
    # Carrega o modelo salvo
    loaded_model = Model.load(save_path)
    
    # Compara as predições do modelo original e do modelo carregado para garantir que ambos sejam idênticos
    print("Testando modelo carregado:")
    for i, inp in enumerate(inputs):
        input_vector = X[i:i+1]
        predicted_word_original = tokenizer.predict(model.predict(input_vector))
        predicted_word_loaded = tokenizer.predict(loaded_model.predict(input_vector))
        print(f"Entrada: {inp}, Previsão (modelo original): {predicted_word_original}, Previsão (modelo carregado): {predicted_word_loaded}, Esperado: {outputs[i]}")
        assert predicted_word_loaded == outputs[i], (
            f"Erro na predição: entrada '{inp}' esperado '{outputs[i]}' obtido '{predicted_word_loaded}'"
        )
    
    # Remove o arquivo do modelo após o teste
    if os.path.exists(save_path):
        os.remove(save_path)

if __name__ == "__main__":
    test_rock_paper_scissors_model()
    print("Teste de Pedra, Papel, Tesoura passou!") 