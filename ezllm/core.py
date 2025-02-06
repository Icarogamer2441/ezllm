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

    def fit(self, X, y, epochs=10, lr=0.01, loss_fn=None, verbose=1, 
            progress_bar=False, test_train=False, 
            output_test_train=False, output_interval=1,  # Intervalo padrão alterado para 1
            tokenizer=None, output_path="training", img_size=None):  # Dependências para geração
        """
        Treina o modelo.
        loss_fn deve ser uma função que receba (y_pred, y_true) e retorne:
            loss, grad_loss
        progress_bar: se True, exibe uma barra de progresso simples durante o treinamento.
        """
        # Validação dos novos parâmetros
        if output_test_train:
            if not tokenizer or not img_size:
                raise ValueError("Para output_test_train=True, tokenizer e img_size devem ser fornecidos")
        
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

            # Cálculo do teste durante o treino
            test_loss = None
            if test_train:
                test_output = self.forward(X)
                test_loss, _ = loss_fn(test_output, y)
            
            # Exibição do progresso
            if verbose or (progress_bar and epoch % 10 == 0):
                msg = f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}"
                if test_loss is not None:
                    msg += f" - Test Loss: {test_loss:.4f}"
                print(msg)

            # Geração de imagem durante o treino
            if output_test_train and (epoch % output_interval == 0 or epoch == epochs-1):
                try:
                    generated = self.predict(X)
                    save_path = output_path  # Usa o nome exato fornecido
                    tokenizer.save_image(generated, save_path, img_size)
                    print(f" | Imagem atualizada: {save_path}")
                except Exception as e:
                    print(f"Erro ao gerar imagem: {str(e)}")

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