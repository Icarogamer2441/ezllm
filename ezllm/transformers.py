import numpy as np
from ezllm.layers import Dense, ReLU
from ezllm.attentions import MultiHeadAttention

# Bloco Transformer simples: atenção seguida por rede feed-forward com conexões residuais
class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, causal_mask=True):
        # Camada de atenção multi-head
        self.mha = MultiHeadAttention(num_heads=num_heads, dropout_rate=dropout_rate, causal_mask=causal_mask)
        # Rede feed-forward: Dense -> ReLU -> Dense
        self.ff1 = Dense(d_model, d_ff)
        self.relu = ReLU()
        self.ff2 = Dense(d_ff, d_model)
    
    def forward(self, x):
        # Passo 1: atenção com conexão residual
        attn_out = self.mha.forward(x)
        x = x + attn_out  # skip connection

        # Passo 2: rede feed-forward com conexão residual
        ff_out = self.ff2.forward(self.relu.forward(self.ff1.forward(x)))
        x = x + ff_out  # skip connection
        return x

    def backward(self, grad_output):
        # Backward simplificado: propaga o gradiente pela rede feed-forward e pela camada de atenção, acumulando com o caminho residual.
        # O ramo feed-forward possui:
        #   a = ff1.forward(x)   --> shape: (..., d_ff)
        #   b = ReLU.forward(a)    --> shape: (..., d_ff)
        #   c = ff2.forward(b)     --> shape: (..., d_model)
        #
        # Durante o backward, para um gradiente grad_output (shape: (..., d_model)) vindo da soma:
        #   grad_c = grad_output
        #   grad_b = ff2.backward(grad_c)         --> shape: (..., d_ff)
        #   grad_a = ReLU.backward(grad_b)          --> shape: (..., d_ff)
        #   grad_ff = ff1.backward(grad_a)          --> shape: (..., d_model)
        grad_c = grad_output
        grad_b = self.ff2.backward(grad_c)
        grad_a = self.relu.backward(grad_b)
        grad_ff = self.ff1.backward(grad_a)

        grad_attn = self.mha.backward(grad_output)
        # Soma o gradiente proveniente do feed-forward, da atenção e do caminho residual direto
        return grad_ff + grad_attn + grad_output

    def update(self, lr):
        self.ff1.update(lr)
        self.ff2.update(lr)
        self.mha.update(lr)

    @property
    def parameters(self):
        return self.ff1.parameters + self.ff2.parameters

# Modelo Transformer completo
class Transformer:
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size):
        """
        Inicializa um modelo Transformer.

        Args:
            num_layers (int): Número de camadas/blocos Transformer no modelo. 
                             Cada bloco contém uma camada de atenção e uma rede feed-forward.
                             Mais camadas permitem que o modelo aprenda padrões mais complexos,
                             mas também aumentam o tempo de treinamento e uso de memória.

            d_model (int): Dimensão dos vetores de embedding (representação dos tokens).
                          Define o tamanho dos vetores que representam cada token ao longo
                          de todo o modelo. Valores maiores permitem representações mais ricas,
                          mas aumentam o custo computacional.

            num_heads (int): Número de "cabeças" de atenção no mecanismo de atenção multi-head.
                            Cada cabeça aprende a focar em diferentes aspectos da sequência.
                            Mais cabeças permitem que o modelo aprenda diferentes tipos de
                            relações entre os tokens, mas também aumentam o custo computacional.

            d_ff (int): Dimensão da camada interna da rede feed-forward dentro de cada bloco Transformer.
                       Define o tamanho da camada oculta na rede neural que processa a saída
                       da camada de atenção. Valores maiores permitem maior capacidade de
                       aprendizado, mas aumentam o custo computacional.

            vocab_size (int): Tamanho do vocabulário (número de tokens únicos que o modelo pode processar).
                             Define quantos tokens diferentes o modelo pode reconhecer e gerar.
                             Deve corresponder ao tamanho do vocabulário usado pelo tokenizador.
        """
        self.vocab_size = vocab_size
        # Camada de embedding: converte vetor one-hot (dim = vocab_size) em vetor d_model
        self.embedding = Dense(vocab_size, d_model)
        # Empilha vários blocos transformer
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        # Camada de saída: projeta de volta para vocab_size
        self.output_layer = Dense(d_model, vocab_size)
    
    def forward(self, x):
        # x: entrada one-hot com shape (..., vocab_size)
        x_embed = self.embedding.forward(x)
        for block in self.blocks:
            x_embed = block.forward(x_embed)
        logits = self.output_layer.forward(x_embed)
        return logits

    def backward(self, grad_loss):
        grad = self.output_layer.backward(grad_loss)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        grad = self.embedding.backward(grad)
        return grad

    def update(self, lr):
        self.embedding.update(lr)
        for block in self.blocks:
            block.update(lr)
        self.output_layer.update(lr)

    def fit(self, X, y, epochs=100, lr=0.01, loss_fn=None, verbose=True):
        """
        Treina o modelo Transformer.
        
        Args:
            X: Dados de entrada com shape (batch, seq_len, vocab_size)
            y: Dados de saída com shape (batch, seq_len, vocab_size)
            epochs: Número de épocas para treinamento.
            lr: Taxa de aprendizado.
            loss_fn: Função de loss que retorna (loss, grad_loss). Ex.: cross_entropy_loss.
            verbose: Se True, imprime atualizações durante o treinamento.
        """
        if loss_fn is None:
            raise ValueError("É necessário fornecer uma função de loss para o treinamento (ex.: cross_entropy_loss).")

        for epoch in range(epochs):
            # Forward pass
            logits = self.forward(X)
            batch, seq_len, _ = logits.shape
            # Reshape para cálculo da loss (2D)
            logits_reshaped = logits.reshape(batch * seq_len, -1)
            y_reshaped = y.reshape(batch * seq_len, -1)
            loss, grad_loss = loss_fn(logits_reshaped, y_reshaped)
            # Restaura a forma original do gradiente
            grad_loss = grad_loss.reshape(batch, seq_len, -1)

            # Backward e atualização
            self.backward(grad_loss)
            self.update(lr)

            if verbose and (epoch % max(1, (epochs // 10)) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

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
        Dada a sequência de tokens codificada (one-hot), retorna os índices dos próximos tokens previstos.
        
        Args:
            input_sequence: array com shape (batch, seq_len, vocab_size)
            temperature: parâmetro para suavização dos logits (default=1.0)
        
        Retorna:
            next_token_idx: array com shape (batch,) contendo os índices do próximo token.
        """
        logits = self.forward(input_sequence)
        last_logits = logits[:, -1, :]  # shape: (batch, vocab_size)
        scaled_logits = last_logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        next_token_idx = np.argmax(probs, axis=1)
        return next_token_idx

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
        Ajusta o tokenizador utilizando os textos fornecidos.
        Se 'texts' for uma string, converte-a para uma lista.
        """
        if isinstance(texts, str):
            texts = [texts]
        tokens = []
        for text in texts:
            if self.lower:
                text = text.lower()
            if self.sep:
                tokens.extend(text.split(self.sep))
            else:
                tokens.append(text)
        unique_tokens = list(dict.fromkeys(tokens))
        self.word2idx = {token: idx for idx, token in enumerate(unique_tokens)}
        self.idx2word = {idx: token for token, idx in self.word2idx.items()}

    def encode(self, text):
        """
        Codifica um texto em uma sequência de vetores one-hot.
        
        Args:
            text: pode ser uma string ou uma lista de tokens.
        
        Retorna:
            Um array numpy de shape (n_tokens, vocab_size)
        """
        if isinstance(text, str):
            tokens = text.lower().split(self.sep) if self.lower and self.sep else [text]
        else:
            tokens = text
        vocab_size = len(self.word2idx)
        encoded = []
        for token in tokens:
            vec = np.zeros(vocab_size)
            if token in self.word2idx:
                vec[self.word2idx[token]] = 1
            encoded.append(vec)
        return np.array(encoded)

    def decode(self, onehots):
        """
        Decodifica uma sequência de vetores one-hot de volta para tokens.
        
        Args:
            onehots: array de shape (n_tokens, vocab_size)
        
        Retorna:
            Uma lista de tokens.
        """
        tokens = []
        for vec in onehots:
            idx = int(np.argmax(vec))
            token = self.idx2word.get(idx, "<unk>")
            tokens.append(token)
        return tokens

    def encode_onehot(self, text):
        """
        Codifica um texto em uma sequência de vetores one-hot.
        Mantido para compatibilidade com a API do Tokenizer original.
        
        Args:
            text: pode ser uma string ou uma lista de tokens.
        
        Retorna:
            Um array numpy de shape (n_tokens, vocab_size)
        """
        return self.encode(text)

def ensure_3d(x):
    """
    Garante que o tensor de entrada tenha 3 dimensões (batch, seq_len, features).
    Se necessário, adiciona dimensões extras.
    
    Args:
        x: Tensor de entrada (numpy array)
    
    Returns:
        Tensor com 3 dimensões
    """
    if x.ndim == 1:
        # Adiciona dimensões de batch e sequência
        return np.expand_dims(np.expand_dims(x, axis=0), axis=0)
    elif x.ndim == 2:
        # Adiciona dimensão de batch
        return np.expand_dims(x, axis=0)
    elif x.ndim == 3:
        # Já está no formato correto
        return x
    else:
        raise ValueError(f"Entrada deve ter 1, 2 ou 3 dimensões, mas recebeu {x.ndim}") 

def create_onehot(index, vocab_size):
    """
    Cria um vetor one-hot automaticamente com o formato correto.
    
    Args:
        index: Índice do token (int)
        vocab_size: Tamanho do vocabulário (int)
    
    Returns:
        Vetor one-hot com shape (1, 1, vocab_size)
    """
    onehot = np.zeros((1, 1, vocab_size))
    onehot[0, 0, index] = 1
    return onehot 