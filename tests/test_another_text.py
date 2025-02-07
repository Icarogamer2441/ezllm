import numpy as np
from ezllm.tokenizer import Tokenizer
from ezllm.core import Model
from ezllm.layers import Dense, ReLU
from ezllm.losses import cross_entropy_loss

def test_fit_another_text():
    # Cria o tokenizador e ajusta um vocabulário inicial com duas palavras
    tokenizer = Tokenizer(lower=True, sep=" ")
    initial_texts = ["hello", "world"]
    tokenizer.fit(initial_texts)
    initial_vocab_size = tokenizer.vocab_size()
    assert initial_vocab_size == 2, f"Vocabulário inicial deve ter 2 tokens, mas encontrou {initial_vocab_size}"
    
    # Adiciona novas palavras ao vocabulário sem sobrescrever
    tokenizer.fit_another_text("goodbye everyone")
    new_vocab_size = tokenizer.vocab_size()
    # Espera-se que "goodbye" e "everyone" sejam adicionados, totalizando 4 tokens
    assert new_vocab_size == 4, f"Vocabulário deve ter 4 tokens após a atualização, mas encontrou {new_vocab_size}"

    # Cria um dataset simples para treinamento: o objetivo é aprender o mapeamento identidade
    # Para simplificar, trabalhamos com tokens únicos (uma palavra por exemplo)
    X_list = ["goodbye", "everyone"]
    y_list = ["goodbye", "everyone"]

    # Constrói um modelo simples com dimensões ajustadas ao novo tamanho do vocabulário (4)
    model = Model([
        Dense(4, 8),
        ReLU(),
        Dense(8, 4)
    ])

    # Treina o modelo com os novos dados de texto usando fit_another_text
    model.fit_another_text(
        X_list, y_list, tokenizer,
        epochs=100, lr=0.1, loss_fn=cross_entropy_loss, verbose=0
    )

    # Testa as predições para verificar se o fluxo está funcionando corretamente
    for token in X_list:
        # Codifica o token individualmente
        encoded = tokenizer.encode_onehot(token)
        # O modelo espera um batch, então envolve o vetor num array de dimensão 2
        pred_logits = model.predict(np.array([encoded]), temperature=1.0)
        predicted_token = tokenizer.predict(pred_logits)
        # Apenas verifica se a palavra predita está no vocabulário (integração de fit e predict)
        assert predicted_token in tokenizer.word2idx, f"Token previsto '{predicted_token}' não está no vocabulário"

def test_training_rounds():
    """
    Treina o modelo em duas rodadas:
    • A primeira rodada utiliza o método tradicional (model.fit) com dados já codificados.
    • A segunda rodada realiza o re-treinamento utilizando fit_another_text com dados em texto cru.
    """
    # Cria o tokenizador e ajusta o vocabulário inicial com duas palavras
    tokenizer = Tokenizer(lower=True, sep=" ")
    initial_texts = ["hello", "world"]
    tokenizer.fit(initial_texts)  # Vocabulário: {"hello": 0, "world": 1}
    vocab_size = tokenizer.vocab_size()  # Deve ser 2
    
    # Cria um modelo simples com dimensões compatíveis com o vocabulário (input e output de tamanho 2)
    model = Model([
        Dense(vocab_size, 4),
        ReLU(),
        Dense(4, vocab_size)
    ])
    
    # Primeira rodada de treinamento usando model.fit com dados já codificados
    X1 = "hello"
    y1 = "hello"
    X1_encoded = tokenizer.encode_onehot(X1)
    # Se houver apenas um token, force para batch 2D
    if X1_encoded.ndim == 1:
        X1_encoded = np.array([X1_encoded])
        y1_encoded = np.array([tokenizer.encode_onehot(y1)])
    else:
        y1_encoded = tokenizer.encode_onehot(y1)
    
    model.fit(X1_encoded, y1_encoded, epochs=50, lr=0.1, loss_fn=cross_entropy_loss, verbose=0)
    
    # Segunda rodada de treinamento usando fit_another_text com dados de texto cru
    X2 = "world"
    y2 = "world"
    model.fit_another_text(X2, y2, tokenizer, epochs=50, lr=0.1, loss_fn=cross_entropy_loss, verbose=0)
    
    # Verifica as predições para ambas as palavras do vocabulário
    for token in ["hello", "world"]:
        encoded = tokenizer.encode_onehot(token)
        if encoded.ndim == 1:
            encoded = np.array([encoded])
        pred_logits = model.predict(encoded, temperature=1.0)
        predicted_token = tokenizer.predict(pred_logits)
        assert predicted_token in tokenizer.word2idx, f"Token previsto '{predicted_token}' não está no vocabulário"
    

if __name__ == "__main__":
    test_fit_another_text()
    test_training_rounds()
    print("Todos os testes fit_another_text e de treinamento por duas rodadas passaram!")
