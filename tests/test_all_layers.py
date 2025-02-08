import numpy as np
import time
from ezllm.layers import Dense, ReLU, Dropout, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Linear
from ezllm.core import Model
from ezllm.tokenizer import Tokenizer
from ezllm.losses import cross_entropy_loss

def create_dataset():
    # Cria um dataset simples para completação de texto
    texts = [
        "eu gosto de programar",
        "python é uma linguagem poderosa",
        "redes neurais são incríveis",
        "aprendizado profundo é fascinante",
        "inteligência artificial está em todo lugar"
    ]
    tokenizer = Tokenizer(lower=True, sep=" ")
    tokenizer.fit(texts)
    
    # Cria pares de input e target
    X = []
    y = []
    for text in texts:
        tokens = text.split()
        for i in range(1, len(tokens)):
            X.append(" ".join(tokens[:i]))
            y.append(tokens[i])
    
    # Codifica em one-hot
    X_onehot = tokenizer.encode_onehot(X)
    y_onehot = tokenizer.encode_onehot(y)
    
    return X_onehot, y_onehot, tokenizer

def calculate_accuracy(model, X, y, tokenizer):
    """Calcula a precisão do modelo de forma mais robusta"""
    correct = 0
    total = len(X)
    
    for i in range(total):
        pred = model.predict(X[i:i+1])
        predicted_token = tokenizer.predict(pred)
        true_token = tokenizer.decode(y[i])
        
        # Verifica se a predição está correta
        if predicted_token == true_token:
            correct += 1
            
    return correct / total

def test_layer(layer_class, input_dim, output_dim, X, y, tokenizer):
    # Cria o modelo com a layer específica
    layers = [
        Dense(input_dim, 64),
        ReLU(),
        layer_class,
        Dense(64, output_dim)
    ]
    model = Model(layers)
    
    # Treina o modelo
    start_time = time.time()
    model.fit(X, y, epochs=100, lr=0.01, loss_fn=cross_entropy_loss, verbose=0)
    training_time = time.time() - start_time
    
    # Avalia a precisão
    correct = 0
    for i in range(len(X)):
        pred = model.predict(X[i:i+1])
        predicted_token = tokenizer.predict(pred)
        if predicted_token == tokenizer.decode(y[i]):
            correct += 1
    accuracy = correct / len(X)
    
    return training_time, accuracy

def test_all_layers():
    X, y, tokenizer = create_dataset()
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    
    # Lista de layers para testar (apenas 2D)
    layers_to_test = [
        ("Dense", Dense(64, 64)),
        ("ReLU", ReLU()),
        ("Dropout", Dropout(dropout_rate=0.2)),
        ("BatchNormalization", BatchNormalization(num_features=64)),
        ("Linear", Linear(64, 64))
    ]
    
    # Testa cada layer
    results = {}
    for name, layer in layers_to_test:
        print(f"Testando layer: {name}")
        try:
            training_time, accuracy = test_layer(layer, input_dim, output_dim, X, y, tokenizer)
            results[name] = {
                "training_time": training_time,
                "accuracy": accuracy
            }
            print(f"{name}: Tempo de treinamento = {training_time:.2f}s, Precisão = {accuracy:.2%}")
        except Exception as e:
            print(f"Erro ao testar {name}: {str(e)}")
            results[name] = {
                "training_time": float('inf'),
                "accuracy": 0.0
            }

if __name__ == "__main__":
    test_all_layers() 