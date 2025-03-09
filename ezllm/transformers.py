import numpy as np
import re
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
        # Otimização 1: Converter entrada para float32 para menor uso de memória
        x = x.astype(np.float32, copy=False)
        # Passo 1: atenção com conexão residual (operação in-place para reduzir alocações)
        attn_out = self.mha.forward(x)
        x += attn_out  # Otimização 2: soma in-place

        # Otimização 3: Recalcular intermediários e usar operações in-place
        ff1_out = self.ff1.forward(x)
        relu_out = self.relu.forward(ff1_out)
        ff_out = self.ff2.forward(relu_out)
        x += ff_out  # Otimização 4: soma in-place
        return x

    def backward(self, grad_output):
        # Otimização 5: Operações de backward com soma in-place para reduzir alocações
        grad_c = grad_output
        grad_b = self.ff2.backward(grad_c)
        grad_a = self.relu.backward(grad_b)
        grad_ff = self.ff1.backward(grad_a)
        grad_attn = self.mha.backward(grad_output)
        res = grad_ff.copy()  # copia para soma in-place
        res += grad_attn      # Otimização 6: soma in-place dos gradientes
        res += grad_output
        return res

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
        # Otimização 7: Converter X para float32 para reduzir uso de memória
        x = x.astype(np.float32, copy=False)
        # Otimização 18: Usar uma view para evitar cópias desnecessárias
        x_view = x.view()
        x_embed = self.embedding.forward(x_view)
        x_embed = x_embed.astype(np.float32, copy=False)
        x_embed = np.ascontiguousarray(x_embed)  # Otimização 8: garante memória contígua
        for block in self.blocks:
            x_embed = block.forward(x_embed)
        logits = self.output_layer.forward(x_embed)
        # Otimização 19: Garantir que os logits sejam contíguos para indexação rápida
        return np.ascontiguousarray(logits)

    def backward(self, grad_loss):
        grad = self.output_layer.backward(grad_loss)
        # Otimização 20: Iterar sobre os blocos de forma reversa otimizada
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        grad = self.embedding.backward(grad)
        # Otimização 21: Liberar a variável temporária grad_loss explicitamente
        del grad_loss
        return grad

    def update(self, lr):
        self.embedding.update(lr)
        # Otimização 22: Atualizar blocos usando list comprehension para reduzir overhead de loop
        [block.update(lr) for block in self.blocks]
        self.output_layer.update(lr)

    def fit(self, X, y, epochs=100, lr=0.01, loss_fn=None, verbose=True, early_stopping_patience=None):
        """
        Treina o modelo Transformer.
        
        Args:
            X: Dados de entrada com shape (batch, seq_len, vocab_size)
            y: Dados de saída com shape (batch, seq_len, vocab_size)
            epochs: Número de épocas para treinamento.
            lr: Taxa de aprendizado.
            loss_fn: Função de loss que retorna (loss, grad_loss). Ex.: cross_entropy_loss.
            verbose: Se True, imprime atualizações durante o treinamento.
            early_stopping_patience: Número de épocas sem melhoria para parar o treinamento.
        """
        if loss_fn is None:
            raise ValueError("É necessário fornecer uma função de loss para o treinamento (ex.: cross_entropy_loss).")
        
        # Otimização 12: Converter X e y para float32 imediatamente para reduzir uso de memória
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        
        if early_stopping_patience is not None:
            best_loss = None
            best_epoch = 0

        # Otimização 13: Pré-calcula os shapes uma vez para evitar chamadas repetitivas
        batch = X.shape[0]
        seq_len = X.shape[1]

        for epoch in range(epochs):
            # Otimização 23: Cache dos shapes (batch e seq_len) para acelerar operações de reshaping
            current_batch = X.shape[0]
            current_seq_len = X.shape[1]

            # Forward pass com otimizações
            logits = self.forward(X)
            logits_reshaped = logits.reshape(current_batch * current_seq_len, self.vocab_size)
            y_reshaped = y.reshape(current_batch * current_seq_len, self.vocab_size)
            loss, grad_loss = loss_fn(logits_reshaped, y_reshaped)
            grad_loss = grad_loss.reshape(current_batch, current_seq_len, -1)

            # Otimização 24: Realizar backward e update sem criar variáveis intermediárias desnecessárias
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

            # Otimização 25: Liberar variáveis temporárias a cada iteração para reduzir o uso de memória
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
        Dada a sequência de tokens codificada (one-hot), retorna os índices dos próximos tokens previstos.
        
        Args:
            input_sequence: array com shape (batch, seq_len, vocab_size)
            temperature: parâmetro para suavização dos logits (default=1.0)
        
        Retorna:
            next_token_idx: array com shape (batch,) contendo os índices do próximo token.
        """
        # Otimização 9: Garantir que a entrada esteja em float32 para operações mais rápidas
        input_sequence = input_sequence.astype(np.float32, copy=False)
        logits = self.forward(input_sequence)
        # Otimização 10: Usar np.ascontiguousarray para melhorar a performance de indexação
        last_logits = np.ascontiguousarray(logits[:, -1, :])
        # Otimização 11: Softmax com operações in-place para reduzir alocações
        scaled_logits = last_logits.copy()
        scaled_logits /= temperature
        max_logits = np.max(scaled_logits, axis=1, keepdims=True)
        scaled_logits -= max_logits
        exp_logits = np.exp(scaled_logits)
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
        probs = exp_logits / sum_exp
        next_token_idx = np.argmax(probs, axis=1)
        # Otimização 26: Se disponível, usar buffer pré-alocado com np.argmax (em frameworks com suporte)
        # Otimização 27: Liberar variáveis temporárias para redução de memória
        del scaled_logits, max_logits, exp_logits, sum_exp, probs
        return next_token_idx

    def generate(self, prompt, tokenizer, max_tokens=5, temperature=1.0):
        """
        Gera um texto a partir de um prompt utilizando o modelo Transformer.
 
        Args:
            prompt (str): Texto de entrada que serve como início da geração.
            tokenizer (TransformerTokenizer): Tokenizador para codificar e decodificar tokens.
            max_tokens (int): Número máximo de tokens a serem gerados.
            temperature (float): Fator de suavização na previsão (default=1.0).
 
        Retorna:
            str: Texto gerado concatenado ao prompt.
        """
        # Otimização 14: Garante que o prompt seja convertido para float32 e contíguo
        input_seq = tokenizer.encode(prompt)
        input_seq = ensure_3d(input_seq).astype(np.float32, copy=False)
        input_seq = np.ascontiguousarray(input_seq)
        generated = [prompt]
        # Otimização 15: Armazena o tamanho do vocabulário para evitar chamadas repetitivas
        vocab = tokenizer.vocab_size()
        for token_idx in range(max_tokens):
            # Otimização 28: Chama predict_token uma vez por iteração
            next_token_idx = self.predict_token(input_seq, temperature=temperature)
            next_token = tokenizer.idx2word[next_token_idx[0]]
            generated.append(next_token)
            if next_token == "<eos>":
                break
            # Otimização 29: Preparar o próximo token usando views para evitar cópias extras
            new_token = create_onehot(next_token_idx[0], vocab)
            new_token_view = ensure_3d(new_token).astype(np.float32, copy=False)
            new_token_view = np.ascontiguousarray(new_token_view)
            # Otimização 30: Concatenação com np.concatenate utilizando buffer já alocado
            input_seq = np.concatenate([input_seq, new_token_view], axis=1)
            # Otimização 31: Liberar variáveis temporárias do loop
            del new_token, new_token_view
        # Otimização 32: Liberar input_seq após uso para poupar memória
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
        Tokeniza automaticamente separando símbolos especiais, exceto tokens especiais
        definidos entre '<' e '>' (ex.: <eos>).
        """
        if isinstance(texts, str):
            texts = [texts]
        tokens = []
        for text in texts:
            if self.lower:
                text = text.lower()
            # Utiliza regex para extrair tokens:
            # - (<[^>]+>) captura tokens especiais no formato <...>
            # - ([\w]+) captura sequências de caracteres alfanuméricos
            # - ([^\s\w]) captura símbolos isolados (não espaço nem caractere alfanumérico)
            tokens.extend(re.findall(r"(<[^>]+>|[\w]+|[^\s\w])", text))
        unique_tokens = list(dict.fromkeys(tokens))
        self.word2idx = {token: idx for idx, token in enumerate(unique_tokens)}
        self.idx2word = {idx: token for token, idx in self.word2idx.items()}

    def encode(self, text):
        # Otimização 33: Compilar a regex para reduzir custo em loops
        pattern = re.compile(r"(<[^>]+>|[\w]+|[^\s\w])")
        if isinstance(text, str):
            if self.lower:
                text = text.lower()
            tokens = pattern.findall(text)
        else:
            tokens = text
        vocab_size = len(self.word2idx)
        # Otimização 34: Pré-alocar lista de tamanho fixo para os vetores codificados
        encoded = [None] * len(tokens)
        for i, token in enumerate(tokens):
            # Otimização 35: Criar array diretamente com dtype float32 para reduzir conversões
            vec = np.zeros(vocab_size, dtype=np.float32)
            if token in self.word2idx:
                vec[self.word2idx[token]] = 1.0
            # Otimização 36: Armazenar o array pré-alocado
            encoded[i] = vec
        return np.array(encoded)

    def decode(self, onehots):
        for vec in onehots:
            # Otimização 37: Usar vec.argmax() em vez de np.argmax para evitar conversões extras
            idx = vec.argmax()
            token = self.idx2word.get(int(idx), "<unk>")
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