import numpy as np
from .tokenizer import Tokenizer
from .layers import Dense

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

    def predict(self, X, temperature=None):
        logits = self.forward(X)
        if temperature is None:
            return logits
        # Import softmax from losses for probability calculation
        from ezllm.losses import softmax
        scaled_logits = logits / temperature
        probs = softmax(scaled_logits)
        preds = []
        for p in probs:
            idx = np.random.choice(len(p), p=p)
            onehot = np.zeros_like(p)
            onehot[idx] = 1
            preds.append(onehot)
        return np.array(preds)

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

    def fit_another(self, X, y, tokenizer, epochs=10, lr=0.01, loss_fn=None, verbose=1,
                   progress_bar=False, test_train=False, output_test_train=False,
                   output_interval=1, output_path="training_another", img_size=None, is_image=False):
        """
        Treina o modelo com novos dados (imagens ou texto).
        Suporta dados em texto bruto ou já codificados (np.ndarray).
        """
        if is_image:
            # Modo imagem
            if not tokenizer.color_mode:
                raise ValueError("O tokenizador deve estar no modo de cor (color_mode=True)")
            X_encoded = np.concatenate([tokenizer.encode_image_onehot(img) for img in X])
            y_encoded = np.concatenate([tokenizer.encode_image_onehot(img) for img in y])
        else:
            # Modo texto: Se X e y já forem numpy arrays, assume dados codificados.
            if isinstance(X, np.ndarray):
                X_encoded = X
            else:
                X_encoded = tokenizer.encode_onehot(X)
                if X_encoded.ndim == 1:
                    X_encoded = np.array([X_encoded])
            if isinstance(y, np.ndarray):
                y_encoded = y
            else:
                y_encoded = tokenizer.encode_onehot(y)
                if y_encoded.ndim == 1:
                    y_encoded = np.array([y_encoded])
        
        # Realiza o treinamento
        self.fit(X_encoded, y_encoded, epochs=epochs, lr=lr, loss_fn=loss_fn, verbose=verbose,
                progress_bar=progress_bar, test_train=test_train,
                output_test_train=output_test_train, output_interval=output_interval,
                tokenizer=tokenizer, output_path=output_path, img_size=img_size)

    def fit_another_image(self, X, y, tokenizer, epochs=10, lr=0.01, loss_fn=None, verbose=1,
                         progress_bar=False, test_train=False, output_test_train=False,
                         output_interval=1, output_path="training_another", img_size=None):
        """
        Treina o modelo com novas imagens sem sobrescrever o vocabulário existente.
        """
        # Verifica se o tokenizador está no modo de cor
        if not tokenizer.color_mode:
            raise ValueError("O tokenizador deve estar no modo de cor (color_mode=True)")

        # Codifica as novas imagens
        X_encoded = np.concatenate([tokenizer.encode_image_onehot(img) for img in X])
        y_encoded = np.concatenate([tokenizer.encode_image_onehot(img) for img in y])

        # Realiza o treinamento
        self.fit(X_encoded, y_encoded, epochs=epochs, lr=lr, loss_fn=loss_fn, verbose=verbose,
                progress_bar=progress_bar, test_train=test_train,
                output_test_train=output_test_train, output_interval=output_interval,
                tokenizer=tokenizer, output_path=output_path, img_size=img_size)

    # Nova função adicionada para textos
    def fit_another_text(self, X, y, tokenizer, epochs=10, lr=0.01, loss_fn=None, verbose=1,
                         progress_bar=False, test_train=False, output_test_train=False,
                         output_interval=1, output_path="training_another", img_size=None):
        """
        Treina o modelo com novos dados de texto sem sobrescrever o vocabulário existente.
        """
        # Internamente, chama fit_another com is_image definido como False para modo texto
        self.fit_another(X, y, tokenizer, epochs=epochs, lr=lr, loss_fn=loss_fn, verbose=verbose,
                         progress_bar=progress_bar, test_train=test_train,
                         output_test_train=output_test_train, output_interval=output_interval,
                         output_path=output_path, img_size=img_size, is_image=False)

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