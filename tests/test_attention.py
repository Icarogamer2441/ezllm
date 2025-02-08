import numpy as np
from ezllm.core import Model
from ezllm.attentions import Attention, Multilayer_perception
from ezllm.layers import Dense, ReLU
from ezllm.tokenizer import Tokenizer
from ezllm.losses import cross_entropy_loss

def test_text_completion(num_attention_layers=1):
    # Configuração do tokenizador e vocabulário
    tokenizer = Tokenizer(lower=True, sep=" ")

    tokens = ["ola", "mundo", "como", "vai", "você", "eu", "gosto", "de", "programar", "em", "python", "por", "que", "ele", "é", "legal"]
    tokenizer.fit(tokens)
    
    X = tokens
    y = tokens[1:] + ["<eos>"]

    X_train_onehot = tokenizer.encode_onehot(X)[np.newaxis, :]
    y_train_onehot = tokenizer.encode_onehot(y)[np.newaxis, :]

    # Configuração do modelo:
    # - Usa duas camadas Dense com ReLU no meio
    # - Adiciona camadas de atenção conforme especificado
    layers = [
        Dense(X_train_onehot.shape[2], 64, num_attention_layers=num_attention_layers),
        ReLU(),
        Dense(64, y_train_onehot.shape[2], num_attention_layers=num_attention_layers)
    ]

    model = Model(layers, attentions=num_attention_layers)

    model.fit(X_train_onehot, y_train_onehot, epochs=100, lr=0.01, loss_fn=cross_entropy_loss, progress_bar=True, verbose=0)

    tests = ["eu", "gosto", "de", "python"]

    print(" ".join(tests), end=" ")

    for i in range(5):
        token = tokenizer.predict(model.predict(tokenizer.encode_onehot(tests if len(tests) > 1 else "eu")))

        print(token if isinstance(token, str) else token[-1], end=" ")
        if isinstance(token, str):
            tests = tests + [token]
        else:
            tests = tests + token

    print()

    print("test_text_completion passou!")


if __name__ == "__main__":
    test_text_completion(5)