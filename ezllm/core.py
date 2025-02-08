import numpy as np
from .tokenizer import Tokenizer
from .layers import Dense
from ezllm.attentions import MultiHeadAttention

class Model:
    def __init__(self, layers, attentions=None):
        self.layers = layers
        # Se o parâmetro 'attentions' for um inteiro, crie esse número de MultiHeadAttention
        if isinstance(attentions, int):
            self.attentions = [MultiHeadAttention(num_heads=4, dropout_rate=0.1, causal_mask=True)
                                for _ in range(attentions)]
        else:
            self.attentions = attentions if attentions is not None else []
        self.num_attention_layers = len(self.attentions)

    def forward(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        for att in self.attentions:
            a = att.forward(a)
        return a

    def backward(self, grad_loss):
        grad = grad_loss
        for att in reversed(self.attentions):
            grad = att.backward(grad)
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
            output_test_train=False, output_interval=1,
            tokenizer=None, output_path="training", img_size=None):
        """
        Treina o modelo.
        loss_fn deve ser uma função que receba (y_pred, y_true) e retorne:
            loss, grad_loss
        progress_bar: se True, exibe uma barra de progresso simples durante o treinamento.
        """
        # Se for o treinamento inicial, registra o tamanho do vocabulário "base" usado.
        if tokenizer is not None and not hasattr(self, 'base_vocab_size'):
            # Como X é one-hot, a posição ativa (argmax) indica o token utilizado.
            if X.ndim == 2:
                active_indices = set(np.argmax(X, axis=1))
            else:
                active_indices = {np.argmax(X)}
            self.base_vocab_size = max(active_indices) + 1

        if output_test_train:
            if not tokenizer or not img_size:
                raise ValueError("Para output_test_train=True, tokenizer e img_size devem ser fornecidos")
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Calcula o loss
            loss, grad_loss = loss_fn(y_pred, y)
            
            # Verifica se o loss é NaN
            if np.isnan(loss):
                print("Loss se tornou NaN. Parando o treinamento.")
                break
            
            # Backward pass
            self.backward(grad_loss)
            
            # Atualiza os parâmetros
            for layer in self.layers:
                if hasattr(layer, 'update'):
                    layer.update(lr)
            for att in self.attentions:
                if hasattr(att, 'update'):
                    att.update(lr)
            
            # Exibição do progresso
            if verbose or (progress_bar and epoch % 10 == 0):
                msg = f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}"
                if test_train:
                    test_output = self.forward(X)
                    test_loss, _ = loss_fn(test_output, y)
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

        # Armazena os dados base se não estiverem já armazenados
        if tokenizer is not None and not hasattr(self, 'X_base'):
            self.X_base = X.copy()
            self.y_base = y.copy()

    def fit_another(self, X, y, tokenizer, epochs=10, lr=0.01, loss_fn=None, verbose=1,
                   progress_bar=False, test_train=False, output_test_train=False,
                   output_interval=1, output_path="training_another", img_size=None, is_image=False):
        """
        Treina o modelo com novos dados (imagens ou texto) realizando fine tuning,
        de forma a preservar o aprendizado anterior.
        Suporta dados em texto bruto ou já codificados (np.ndarray).
        """
        # Armazena pesos antigos para preservar o aprendizado anterior
        old_weights = []
        for layer in self.layers:
            if hasattr(layer, 'W'):
                old_weights.append({'W': layer.W.copy(), 'b': layer.b.copy()})
            else:
                old_weights.append(None)

        # Define um learning rate reduzido para o fine tuning
        fine_tune_lr = lr * 0.1

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
        
        # Combina os dados incrementais com uma fração dos dados base para evitar o esquecimento
        if hasattr(self, 'X_base'):
            replay_ratio = 0.8  # 80% de replay dos dados base
            num_new = X_encoded.shape[0]
            num_base = int(num_new * replay_ratio)
            num_base = min(num_base, self.X_base.shape[0])  # Garante que não ultrapasse os dados disponíveis
            if num_base > 0:
                X_replay = self.X_base[:num_base]
                y_replay = self.y_base[:num_base]
                X_encoded = np.concatenate((X_replay, X_encoded), axis=0)
                y_encoded = np.concatenate((y_replay, y_encoded), axis=0)

        # Realiza o treinamento com fine tuning (lr reduzido)
        self.fit(X_encoded, y_encoded, epochs=epochs, lr=fine_tune_lr, loss_fn=loss_fn, verbose=verbose,
                progress_bar=progress_bar, test_train=test_train,
                output_test_train=output_test_train, output_interval=output_interval,
                tokenizer=tokenizer, output_path=output_path, img_size=img_size)

        # Blending: preserva fortemente os pesos para os tokens já aprendidos (base)
        default_preservation = 0.1   # Para partes não token‑específicas ou tokens novos
        token_preservation = 0.001   # Para tokens já existentes: preserva 99.9% do peso antigo
        current_vocab = tokenizer.vocab_size()
        base_vocab = self.base_vocab_size if hasattr(self, 'base_vocab_size') else 0

        for i, layer in enumerate(self.layers):
            if old_weights[i] is not None:
                # Se a camada for token‑específica na entrada (por exemplo, a primeira Dense)
                if layer.W.shape[0] == current_vocab:
                    newW = layer.W.copy()
                    for r in range(newW.shape[0]):
                        if r < base_vocab:
                            newW[r, :] = (1 - token_preservation) * old_weights[i]['W'][r, :] + token_preservation * newW[r, :]
                        else:
                            newW[r, :] = (1 - default_preservation) * old_weights[i]['W'][r, :] + default_preservation * newW[r, :]
                    layer.W = newW
                # Se a camada for token‑específica na saída (por exemplo, a última Dense)
                if layer.W.shape[1] == current_vocab:
                    newW = layer.W.copy()
                    for c in range(newW.shape[1]):
                        if c < base_vocab:
                            newW[:, c] = (1 - token_preservation) * old_weights[i]['W'][:, c] + token_preservation * newW[:, c]
                        else:
                            newW[:, c] = (1 - default_preservation) * old_weights[i]['W'][:, c] + default_preservation * newW[:, c]
                    layer.W = newW
                # Bias: se o vetor de bias tiver dimensão igual ao vocabulário (tokens na saída)
                if layer.b.shape[0] == current_vocab:
                    newb = layer.b.copy()
                    for j in range(len(newb)):
                        if j < base_vocab:
                            newb[j] = (1 - token_preservation) * old_weights[i]['b'][j] + token_preservation * newb[j]
                        else:
                            newb[j] = (1 - default_preservation) * old_weights[i]['b'][j] + default_preservation * newb[j]
                    layer.b = newb

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

    def adjust_model_toBetter(self, X, y, tokenizer, epochs=50, lr=0.01, loss_fn=None, verbose=1, **kwargs):
        """
        Realiza um fine-tuning no modelo utilizando todos os dados de treinamento de texto.
        Essa função permite ajustar melhor o modelo com todos os dados, com uma taxa de aprendizado ainda reduzida.
        
        Parâmetros:
          X: Dados de entrada (texto ou array one-hot).
          y: Rótulos correspondentes.
          tokenizer: Tokenizador utilizado.
          epochs: Número de épocas para fine-tuning.
          lr: Taxa de aprendizado (será multiplicada por 0.1 internamente).
          loss_fn: Função de loss.
          verbose: Flag de verbosidade.
          kwargs: Parâmetros adicionais que serão repassados para self.fit().
        """
        fine_tuning_lr = lr * 0.1
        self.fit(X, y, epochs=epochs, lr=fine_tuning_lr, loss_fn=loss_fn, verbose=verbose,
                 tokenizer=tokenizer, **kwargs)

    def adjust_model_toBetter_inImage(self, X, y, tokenizer, epochs=50, lr=0.01, loss_fn=None, verbose=1, img_size=None, **kwargs):
        """
        Realiza um fine-tuning no modelo utilizando todos os dados de treinamento de imagens.
        É necessário que o tokenizador esteja no modo de cor (color_mode=True).
        
        Parâmetros:
          X: Lista de imagens (PIL.Image ou array numpy no formato apropriado).
          y: Imagens alvo para o treinamento.
          tokenizer: Tokenizador configurado para imagens (color_mode=True).
          epochs: Número de épocas para fine-tuning.
          lr: Taxa de aprendizado (será multiplicada por 0.1 internamente).
          loss_fn: Função de loss.
          verbose: Flag de verbosidade.
          img_size: Tamanho original da imagem para salvar (caso necessário para output_test_train).
          kwargs: Parâmetros adicionais que serão repassados para self.fit() ou self.fit_another().
        """
        if not tokenizer.color_mode:
            raise ValueError("Para ajustar imagens, o tokenizador deve estar no modo de cor (color_mode=True)")
        fine_tuning_lr = lr * 0.1
        # Codifica as imagens utilizando o método definido no tokenizador
        X_encoded = np.concatenate([tokenizer.encode_image_onehot(img) for img in X])
        y_encoded = np.concatenate([tokenizer.encode_image_onehot(img) for img in y])
        self.fit(X_encoded, y_encoded, epochs=epochs, lr=fine_tuning_lr, loss_fn=loss_fn, verbose=verbose,
                 tokenizer=tokenizer, img_size=img_size, **kwargs)

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