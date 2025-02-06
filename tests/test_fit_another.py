import numpy as np
from ezllm.core import Model
from ezllm.layers import Dense, ReLU
from ezllm.losses import cross_entropy_loss
from ezllm.tokenizer import Tokenizer

def test_fit_another():
    # Conjunto de dados inicial
    initial_inputs = ["oi", "olá", "tudo", "como", "vai"]
    initial_outputs = ["olá", "oi", "bem", "vai", "bem"]

    # Conjunto de dados adicional para expandir a base de conhecimento
    additional_inputs = ["bom", "dia", "boa", "tarde", "noite", "tchau", "adeus"]
    additional_outputs = ["dia", "bom", "tarde", "boa", "boa", "adeus", "tchau"]

    # Tokenizador treinado com todas as palavras envolvidas
    all_texts = initial_inputs + initial_outputs + additional_inputs + additional_outputs
    tokenizer = Tokenizer(lower=True, sep=" ")
    tokenizer.fit(all_texts)

    # Codificação one-hot dos dados iniciais
    X_initial = tokenizer.encode_onehot(initial_inputs)
    y_initial = tokenizer.encode_onehot(initial_outputs)

    # Codificação one-hot dos dados adicionais
    X_additional = tokenizer.encode_onehot(additional_inputs)
    y_additional = tokenizer.encode_onehot(additional_outputs)

    # Criação do modelo
    model = Model([
        Dense(X_initial.shape[1], 20),  # Camada oculta maior
        ReLU(),
        Dense(20, X_initial.shape[1])
    ])

    # Treinamento inicial
    print("Treinamento inicial...")
    model.fit(X_initial, y_initial, epochs=1000, lr=0.05, loss_fn=cross_entropy_loss, verbose=0)

    # Teste das predições após o treinamento inicial
    print("Testando predições após treinamento inicial:")
    for i, word in enumerate(initial_inputs):
        x = tokenizer.encode_onehot([word])
        predicted_word = tokenizer.predict(model.predict(x))
        expected = initial_outputs[i]
        print(f"Entrada: {word} -> Predição: {predicted_word}, Esperado: {expected}")
        if predicted_word != expected:
            print(f"AVISO: Predição incorreta para '{word}'. Esperado: '{expected}', Obtido: '{predicted_word}'")

    # Fine-tuning com o novo conjunto de dados
    print("\nFine-tuning com fit_another para expandir a base de conhecimento...")
    model.fit_another(X_additional, y_additional, epochs=1000, lr=0.01, loss_fn=cross_entropy_loss, verbose=0)  # Taxa de aprendizado menor para evitar esquecer o aprendizado anterior

    # Teste das predições após o fine-tuning
    print("\nTestando predições após fine-tuning:")
    all_inputs = initial_inputs + additional_inputs
    all_outputs = initial_outputs + additional_outputs
    for i, word in enumerate(all_inputs):
        x = tokenizer.encode_onehot([word])
        predicted_word = tokenizer.predict(model.predict(x))
        expected = all_outputs[i]
        print(f"Entrada: {word} -> Predição: {predicted_word}, Esperado: {expected}")
        if predicted_word != expected:
            print(f"AVISO: Predição incorreta para '{word}'. Esperado: '{expected}', Obtido: '{predicted_word}'")

if __name__ == "__main__":
    test_fit_another() 