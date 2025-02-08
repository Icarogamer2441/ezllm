import numpy as np

class Attention:
    def __init__(self, scale_factor=None, dropout_rate=0.1, causal_mask=False):
        self.input = None
        self.att_weights = None
        self.scale_factor = scale_factor
        self.dropout_rate = dropout_rate
        self.causal_mask = causal_mask
        # Inicializa máscara de dropout
        self.dropout_mask = None

    def forward(self, x):
        self.input = x
        
        # Se a entrada for 2D, adiciona uma dimensão de sequência
        if x.ndim == 2:
            x = x[:, np.newaxis, :]
        
        batch_size, seq_len, input_dim = x.shape
        
        # Calcula scores de atenção
        scale = self.scale_factor if self.scale_factor is not None else np.sqrt(input_dim)
        scores = np.matmul(x, x.transpose(0, 2, 1)) / scale
        
        # Aplica máscara causal se necessário
        if self.causal_mask:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
            scores += mask[None, :, :]
        
        # Aplica softmax
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        att_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Aplica dropout
        if self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*att_weights.shape) > self.dropout_rate) 
            att_weights = att_weights * self.dropout_mask
            att_weights = att_weights / (1 - self.dropout_rate)  # Scale during training
        
        self.att_weights = att_weights
        
        # Aplica os pesos de atenção
        out = x + np.matmul(att_weights, x)
        
        # Remove a dimensão extra se a entrada original era 2D
        if x.ndim == 3 and seq_len == 1:
            out = out.squeeze(1)
        
        return out

    def backward(self, grad_output):
        if self.dropout_rate > 0 and self.dropout_mask is not None:
            # Aplica a máscara de dropout ao gradiente
            # A máscara de dropout é (batch_size, seq_len, seq_len)
            # Precisamos expandir para (batch_size, seq_len, input_dim)
            # Primeiro, aplicamos a máscara ao gradiente que vem da multiplicação matricial
            # O grad_output tem shape (batch_size, seq_len, input_dim)
            # Aplicamos a máscara de dropout apenas na dimensão da sequência
            grad_output = grad_output * self.dropout_mask[:, :, :, np.newaxis].mean(axis=2)
            grad_output = grad_output / (1 - self.dropout_rate)
        return grad_output

    def update(self, lr):
        # Ainda sem parâmetros treináveis
        pass


class Multilayer_perception:
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.W1 = np.random.randn(input_dim, self.hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, output_dim) * np.sqrt(2. / self.hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.input = None

    def forward(self, x):
        self.input = x
        # Camada oculta com função ReLU
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)
        # Segunda transformação linear
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # Conexão residual: soma a entrada original
        out = x + self.z2
        return out

    def backward(self, grad_output):
        # Processa gradientes em mini-batches para reduzir uso de memória
        batch_size = grad_output.shape[0]
        chunk_size = 100  # Processa 100 exemplos por vez
        
        # Inicializa gradientes
        grad_W1 = np.zeros_like(self.W1)
        grad_b1 = np.zeros_like(self.b1)
        grad_W2 = np.zeros_like(self.W2)
        grad_b2 = np.zeros_like(self.b2)
        grad_x = np.zeros_like(self.input)
        
        for i in range(0, batch_size, chunk_size):
            chunk_end = min(i + chunk_size, batch_size)
            grad_chunk = grad_output[i:chunk_end]
            input_chunk = self.input[i:chunk_end]
            a1_chunk = self.a1[i:chunk_end]
            
            # Gradientes para a segunda camada
            grad_W2 += np.dot(a1_chunk.reshape(-1, self.hidden_dim).T, 
                            grad_chunk.reshape(-1, self.output_dim))
            grad_b2 += np.sum(grad_chunk.reshape(-1, self.output_dim), axis=0, keepdims=True)
            grad_a1 = np.dot(grad_chunk, self.W2.T)
            
            # Backpropagate through ReLU
            grad_z1 = grad_a1 * (self.z1[i:chunk_end] > 0).astype(float)
            grad_W1 += np.dot(input_chunk.reshape(-1, self.input_dim).T, 
                            grad_z1.reshape(-1, self.hidden_dim))
            grad_b1 += np.sum(grad_z1.reshape(-1, self.hidden_dim), axis=0, keepdims=True)
            grad_from_mlp = np.dot(grad_z1, self.W1.T)
            
            # Gradiente total para a entrada inclui o skip connection
            grad_x[i:chunk_end] = grad_chunk + grad_from_mlp

        # Normaliza gradientes pelo número de chunks
        num_chunks = np.ceil(batch_size / chunk_size)
        self.grad_W1 = grad_W1 / num_chunks
        self.grad_b1 = grad_b1 / num_chunks
        self.grad_W2 = grad_W2 / num_chunks
        self.grad_b2 = grad_b2 / num_chunks

        return grad_x

    def update(self, lr):
        self.W1 -= lr * self.grad_W1
        self.b1 -= lr * self.grad_b1
        self.W2 -= lr * self.grad_W2
        self.b2 -= lr * self.grad_b2 

# Nova atenção melhorada - MultiHeadAttention
class MultiHeadAttention:
    def __init__(self, num_heads=4, dropout_rate=0.1, causal_mask=False):
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causal_mask = causal_mask
        # Cria num_heads instâncias da atenção simples
        self.heads = [Attention(scale_factor=None, dropout_rate=dropout_rate, causal_mask=causal_mask)
                      for _ in range(num_heads)]
    
    def forward(self, x):
        # Coleta as saídas de cada cabeça
        outs = [head.forward(x.copy()) for head in self.heads]
        # Retorna a média das saídas para manter a mesma dimensão
        out = np.mean(outs, axis=0)
        return out

    def backward(self, grad_output):
        # Propaga o gradiente para cada cabeça e retorna a média
        grad_heads = [head.backward(grad_output.copy()) for head in self.heads]
        return np.mean(grad_heads, axis=0)

    def update(self, lr):
        for head in self.heads:
            head.update(lr) 