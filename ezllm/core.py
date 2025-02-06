import numpy as np

class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, grad_loss):
        grad = grad_loss
        # Propaga o gradiente de trás para frente pelas camadas
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, X):
        return self.forward(X)

    def fit(self, X, y, epochs=10, lr=0.01, loss_fn=None, verbose=1, progress_bar=False):
        """
        Treina o modelo.
        loss_fn deve ser uma função que receba (y_pred, y_true) e retorne:
            loss, grad_loss
        progress_bar: se True, exibe uma barra de progresso simples durante o treinamento.
        """
        for epoch in range(epochs):
            # Propagação direta
            y_pred = self.forward(X)
            # Cálculo da loss e do gradiente associado (backward do loss)
            loss, grad_loss = loss_fn(y_pred, y)
            # Propagação de volta
            self.backward(grad_loss)
            # Atualiza os parâmetros das camadas (se a camada tiver método update)
            for layer in self.layers:
                if hasattr(layer, 'update'):
                    layer.update(lr)
            if progress_bar:
                percent = int((epoch + 1) / epochs * 100)
                bar_length = 20
                num_hashes = int(bar_length * (epoch + 1) / epochs)
                bar = "[" + "#" * num_hashes + "." * (bar_length - num_hashes) + "]"
                print(f"\rEpoch {epoch+1}/{epochs} {bar} {percent}% Loss: {loss:.4f}", end="")
            elif verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        if progress_bar:
            print()  # Nova linha após o progresso

    def save(self, filepath):
        """
        Salva o modelo (incluindo todas as configurações e pesos) em um arquivo.
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Modelo salvo em {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Carrega um modelo previamente salvo a partir de um arquivo.
        """
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Modelo carregado de {filepath}")
        return model 