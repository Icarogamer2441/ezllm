import numpy as np
import re
from ezllm.layers import Dense, ReLU
from ezllm.attentions import MultiHeadAttention

# ===== NOVAS CLASSES PARA MELHORAR O SISTEMA =====

class LayerNormalization:
    """
    Camada de normalização de ativação (Layer Norm) simples.
    """
    def __init__(self, features, epsilon=1e-6):
        self.gamma = np.ones((features,), dtype=np.float32)
        self.beta = np.zeros((features,), dtype=np.float32)
        self.epsilon = epsilon

    def forward(self, x):
        # x: (batch, seq_len, features)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        self.std = np.sqrt(var + self.epsilon)
        self.x_norm = (x - mean) / self.std
        return self.gamma * self.x_norm + self.beta

    def backward(self, grad_output):
        # Implementação simplificada – em produção, use autodiff
        return grad_output

    def update(self, lr):
        # Para esta implementação, os parâmetros não são atualizados via gradientes
        pass

    @property
    def parameters(self):
        return [self.gamma, self.beta]


class PositionalEncoding:
    """
    Codificação posicional com funções senoidais (sinusoidal).
    """
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pos = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)
        self.pe = pe  # shape: (max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]


class ImprovedTransformerBlock:
    """
    Bloco Transformer melhorado com:
     - Normalização prévia (pre-norm)
     - Mecanismo multi-head de atenção
     - Rede feed-forward
     - Conexões residuais
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, causal_mask=True):
        self.layernorm1 = LayerNormalization(d_model)
        self.layernorm2 = LayerNormalization(d_model)
        self.mha = MultiHeadAttention(num_heads=num_heads, dropout_rate=dropout_rate, causal_mask=causal_mask)
        self.ff1 = Dense(d_model, d_ff)
        self.relu = ReLU()
        self.ff2 = Dense(d_ff, d_model)

    def forward(self, x):
        # Pré-norm: normaliza antes da atenção
        norm_x = self.layernorm1.forward(x)
        attn_out = self.mha.forward(norm_x)
        x = x + attn_out  # Conexão residual
        # Nova normalização antes da rede feed-forward
        norm_x2 = self.layernorm2.forward(x)
        ff_intermediate = self.ff1.forward(norm_x2)
        ff_activated = self.relu.forward(ff_intermediate)
        ff_out = self.ff2.forward(ff_activated)
        x = x + ff_out  # Conexão residual
        return x

    def backward(self, grad_output):
        # Backward simplificado – para uma implementação completa, seria necessário detalhar cada operação.
        return grad_output

    def update(self, lr):
        self.mha.update(lr)
        self.ff1.update(lr)
        self.ff2.update(lr)
        # LayerNorm não possui atualização nessa implementação simplificada

    @property
    def parameters(self):
        return self.layernorm1.parameters + self.layernorm2.parameters + \
               self.ff1.parameters + self.ff2.parameters


# ===== MODELAGEM DO TRANSFORMER COMPLETO =====

class Transformer:
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len=512):
        """
        Inicializa um modelo Transformer completo aprimorado.

        Args:
            num_layers (int): Número de blocos Transformer.
            d_model (int): Dimensão dos vetores de embedding.
            num_heads (int): Número de cabeças na atenção multi-head.
            d_ff (int): Dimensão interna da rede feed-forward.
            vocab_size (int): Tamanho do vocabulário.
            max_len (int): Tamanho máximo para codificação posicional.
        """
        self.vocab_size = vocab_size
        # Camada de embedding: transforma vetores one-hot em d_model
        self.embedding = Dense(vocab_size, d_model)
        # Codificação posicional para adicionar informações de posição
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        # Stack com blocos Transformer melhorados (pre‑norm)
        self.blocks = [ImprovedTransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        # Camada de normalização final opcional
        self.layernorm_final = LayerNormalization(d_model)
        # Camada de saída: projeta de d_model para o tamanho do vocabulário
        self.output_layer = Dense(d_model, vocab_size)

    def forward(self, x):
        """
        Executa o forward pass do Transformer.
        Args:
            x: entrada one‑hot com shape (batch, seq_len, vocab_size)
        """
        x = x.astype(np.float32, copy=False)
        # Converte one‑hot em embedding
        x_embed = self.embedding.forward(x)
        x_embed = np.ascontiguousarray(x_embed)
        # Adiciona codificação posicional
        x_embed = self.positional_encoding.forward(x_embed)
        # Passa por cada bloco Transformer melhorado
        for block in self.blocks:
            x_embed = block.forward(x_embed)
        # Normalização final antes da projeção
        x_embed = self.layernorm_final.forward(x_embed)
        logits = self.output_layer.forward(x_embed)
        return np.ascontiguousarray(logits)

    def backward(self, grad_loss):
        grad = self.output_layer.backward(grad_loss)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        grad = self.embedding.backward(grad)
        del grad_loss
        return grad

    def update(self, lr):
        self.embedding.update(lr)
        for block in self.blocks:
            block.update(lr)
        self.output_layer.update(lr)

    def fit(self, X, y, epochs=100, lr=0.01, loss_fn=None, verbose=True, early_stopping_patience=None):
        """
        Treina o modelo Transformer.

        Args:
            X: Dados de entrada (batch, seq_len, vocab_size)
            y: Dados alvo (batch, seq_len, vocab_size)
            epochs (int): Número de épocas de treinamento.
            lr (float): Taxa de aprendizado.
            loss_fn: Função de loss que retorna (loss, grad_loss).
            verbose (bool): Se True, imprime atualizações durante o treinamento.
            early_stopping_patience (int): Épocas sem melhora para parada antecipada.
        """
        if loss_fn is None:
            raise ValueError("Forneça uma função de loss (ex.: cross_entropy_loss).")
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)

        if early_stopping_patience is not None:
            best_loss = None
            best_epoch = 0

        batch = X.shape[0]
        seq_len = X.shape[1]

        for epoch in range(epochs):
            current_batch = X.shape[0]
            current_seq_len = X.shape[1]

            logits = self.forward(X)
            logits_reshaped = logits.reshape(current_batch * current_seq_len, self.vocab_size)
            y_reshaped = y.reshape(current_batch * current_seq_len, self.vocab_size)
            loss, grad_loss = loss_fn(logits_reshaped, y_reshaped)
            grad_loss = grad_loss.reshape(current_batch, current_seq_len, -1)

            self.backward(grad_loss)
            self.update(lr)

            if verbose and (epoch % max(1, (epochs // 10)) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

            if early_stopping_patience is not None:
                if best_loss is None or loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                elif epoch - best_epoch >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

            del logits, logits_reshaped, y_reshaped, grad_loss

        return self

    @property
    def parameters(self):
        params = self.embedding.parameters
        for block in self.blocks:
            params += block.parameters
        params += self.output_layer.parameters
        return params

    def predict_token(self, input_sequence, temperature=1.0):
        """
        Dada uma sequência one-hot, prevê o próximo token.
        Args:
            input_sequence: (batch, seq_len, vocab_size)
            temperature: fator de suavização dos logits (default=1.0)
        Retorna:
            next_token_idx: array (batch,) com índices previstos
        """
        input_sequence = input_sequence.astype(np.float32, copy=False)
        logits = self.forward(input_sequence)
        last_logits = np.ascontiguousarray(logits[:, -1, :])
        scaled_logits = last_logits.copy()
        scaled_logits /= temperature
        max_logits = np.max(scaled_logits, axis=1, keepdims=True)
        scaled_logits -= max_logits
        exp_logits = np.exp(scaled_logits)
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
        probs = exp_logits / sum_exp
        next_token_idx = np.argmax(probs, axis=1)
        del scaled_logits, max_logits, exp_logits, sum_exp, probs
        return next_token_idx

    def generate(self, prompt, tokenizer, max_tokens=5, temperature=1.0):
        """
        Gera texto a partir de um prompt utilizando o modelo Transformer.
 
        Args:
            prompt (str): Texto inicial.
            tokenizer (TransformerTokenizer): Para codificar/decodificar tokens.
            max_tokens (int): Quantidade máxima de tokens a gerar.
            temperature (float): Fator de suavização (default=1.0).
 
        Retorna:
            str: Texto gerado concatenado ao prompt.
        """
        input_seq = tokenizer.encode(prompt)
        input_seq = ensure_3d(input_seq).astype(np.float32, copy=False)
        input_seq = np.ascontiguousarray(input_seq)
        generated = [prompt]
        vocab = tokenizer.vocab_size()
        for _ in range(max_tokens):
            next_token_idx = self.predict_token(input_seq, temperature=temperature)
            next_token = tokenizer.idx2word[next_token_idx[0]]
            generated.append(next_token)
            if next_token == "<eos>":
                break
            new_token = create_onehot(next_token_idx[0], vocab)
            new_token_view = ensure_3d(new_token).astype(np.float32, copy=False)
            new_token_view = np.ascontiguousarray(new_token_view)
            input_seq = np.concatenate([input_seq, new_token_view], axis=1)
            del new_token, new_token_view
        del input_seq
        return " ".join(generated)

    def save(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Modelo Transformer salvo em {filepath}")

    @staticmethod
    def load(filepath):
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Modelo Transformer carregado de {filepath}")
        return model

# ===== TOKENIZER E FUNÇÕES AUXILIARES =====

class TransformerTokenizer:
    def __init__(self, lower=True, sep=" "):
        self.lower = lower
        self.sep = sep
        self.word2idx = {}
        self.idx2word = {}

    def vocab_size(self):
        """Retorna o tamanho do vocabulário"""
        return len(self.word2idx)

    def fit(self, texts):
        """
        Ajusta o tokenizador com os textos fornecidos.
        Tokeniza separando símbolos especiais, exceto tokens definidos entre '<' e '>'.
        """
        if isinstance(texts, str):
            texts = [texts]
        tokens = []
        for text in texts:
            if self.lower:
                text = text.lower()
            # Regex para tokens: tokens especiais (<...>), palavras ou símbolos isolados.
            tokens.extend(re.findall(r"(<[^>]+>|[\w]+|[^\s\w])", text))
        unique_tokens = list(dict.fromkeys(tokens))
        self.word2idx = {token: idx for idx, token in enumerate(unique_tokens)}
        self.idx2word = {idx: token for token, idx in self.word2idx.items()}

    def encode(self, text):
        # Otimização: compilação da regex
        pattern = re.compile(r"(<[^>]+>|[\w]+|[^\s\w])")
        if isinstance(text, str):
            if self.lower:
                text = text.lower()
            tokens = pattern.findall(text)
        else:
            tokens = text
        vocab_size = len(self.word2idx)
        encoded = [None] * len(tokens)
        for i, token in enumerate(tokens):
            vec = np.zeros(vocab_size, dtype=np.float32)
            if token in self.word2idx:
                vec[self.word2idx[token]] = 1.0
            encoded[i] = vec
        return np.array(encoded)

    def decode(self, onehots):
        tokens = []
        for vec in onehots:
            idx = vec.argmax()
            token = self.idx2word.get(int(idx), "<unk>")
            tokens.append(token)
        return tokens

    def encode_onehot(self, text):
        """Codifica um texto em uma sequência de vetores one-hot."""
        return self.encode(text)


def ensure_3d(x):
    """
    Garante que o tensor de entrada tenha 3 dimensões (batch, seq_len, features).
    """
    if x.ndim == 1:
        return np.expand_dims(np.expand_dims(x, axis=0), axis=0)
    elif x.ndim == 2:
        return np.expand_dims(x, axis=0)
    elif x.ndim == 3:
        return x
    else:
        raise ValueError(f"Entrada deve ter 1, 2 ou 3 dimensões, recebeu {x.ndim}")


def create_onehot(index, vocab_size):
    """
    Cria um vetor one-hot com shape (1, 1, vocab_size).
    """
    onehot = np.zeros((1, 1, vocab_size), dtype=np.float32)
    onehot[0, 0, index] = 1.0
    return onehot 