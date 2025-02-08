import numpy as np

class Dense:
    def __init__(self, in_features, out_features, num_attention_layers=0):
        # Inicializa os pesos com uma distribuição normal escalonada
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.b = np.zeros((1, out_features))
        # Variáveis para armazenar os dados necessários no backward
        self.input = None
        self.dW = None
        self.db = None
        self.num_attention_layers = num_attention_layers

    def forward(self, x):
        # Armazena a entrada para uso no backward
        self.input = x
        
        # Se houver camadas de atenção e a entrada for 3D, reshape para (batch_size * seq_len, input_dim)
        if self.num_attention_layers > 0 and x.ndim == 3:
            batch_size, seq_len, input_dim = x.shape
            x_reshaped = x.reshape(-1, input_dim)
            # Aplica a transformação linear
            out = np.dot(x_reshaped, self.W) + self.b
            # Reshape de volta para (batch_size, seq_len, output_dim)
            return out.reshape(batch_size, seq_len, -1)
        else:
            # Para entradas 2D, aplica a transformação linear diretamente
            return np.dot(x, self.W) + self.b

    def backward(self, grad_output):
        # Verifica se a entrada tem 3 dimensões
        if self.input.ndim == 3:
            batch_size, seq_len, input_dim = self.input.shape
            # Reestrutura (flatten) a entrada e o gradiente para 2D
            input_reshaped = self.input.reshape(batch_size * seq_len, input_dim)
            grad_output_reshaped = grad_output.reshape(batch_size * seq_len, -1)
            # Calcula os gradientes
            self.dW = np.dot(input_reshaped.T, grad_output_reshaped)
            self.db = np.sum(grad_output_reshaped, axis=0, keepdims=True)
            # Calcula o gradiente em relação à entrada e volta à forma original
            grad_input = np.dot(grad_output_reshaped, self.W.T)
            grad_input = grad_input.reshape(batch_size, seq_len, input_dim)
            return grad_input
        else:
            self.dW = np.dot(self.input.T, grad_output)
            self.db = np.sum(grad_output, axis=0, keepdims=True)
            grad_input = np.dot(grad_output, self.W.T)
            return grad_input

    def update(self, lr):
        # Atualiza os parâmetros usando descida de gradiente
        self.W -= lr * self.dW
        self.b -= lr * self.db

    @property
    def parameters(self):
        # Retorna os parâmetros e seus gradientes para uso em otimizadores externos
        return [(self.W, self.dW), (self.b, self.db)]

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # Cria uma máscara para os valores positivos
        self.mask = (x > 0).astype(float)
        return x * self.mask

    def backward(self, grad_output):
        # Propaga o gradiente, zerando onde x era negativo ou zero
        return grad_output * self.mask 

# Novos tipos de layers adicionados:

class Dropout:
    def __init__(self, dropout_rate=0.5, training=True):
        self.dropout_rate = dropout_rate
        self.training = training
        self.mask = None

    def forward(self, x):
        # Se não estiver treinando ou a taxa de dropout for zero, retorna a entrada sem modificação
        if not self.training or self.dropout_rate == 0:
            return x
        # Cria uma máscara com probabilidade de manter cada unidade = 1 - dropout_rate e a escala apropriadamente
        self.mask = (np.random.rand(*x.shape) > self.dropout_rate) / (1 - self.dropout_rate)
        return x * self.mask

    def backward(self, grad_output):
        if not self.training or self.dropout_rate == 0:
            return grad_output
        return grad_output * self.mask


class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        # Preserva a dimensão do batch e "achata" o restante
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)


class Conv2D:
    def __init__(self, num_filters, kernel_size, stride=1, padding='same'):
        self.num_filters = num_filters
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = None  # Serão inicializados no primeiro forward
        self.b = None
        self.x = None
        self.x_padded = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        N, H, W, C = x.shape
        KH, KW = self.kernel_size
        if self.padding == 'same':
            pad_h = (KH - 1) // 2
            pad_w = (KW - 1) // 2
        else:
            pad_h, pad_w = 0, 0
        x_padded = np.pad(x, ((0,0), (pad_h, pad_h), (pad_w, pad_w), (0,0)), mode='constant')
        self.x_padded = x_padded
        # Inicializa os pesos na primeira passagem
        if self.W is None:
            scale = np.sqrt(2.0 / (KH * KW * C))
            self.W = np.random.randn(KH, KW, C, self.num_filters) * scale
            self.b = np.zeros((self.num_filters,))
        H_out = (H + 2*pad_h - KH) // self.stride + 1
        W_out = (W + 2*pad_w - KW) // self.stride + 1
        out = np.zeros((N, H_out, W_out, self.num_filters))
        for n in range(N):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * self.stride
                    h_end = h_start + KH
                    w_start = j * self.stride
                    w_end = w_start + KW
                    x_slice = x_padded[n, h_start:h_end, w_start:w_end, :]
                    for f in range(self.num_filters):
                        out[n, i, j, f] = np.sum(x_slice * self.W[..., f]) + self.b[f]
        return out

    def backward(self, grad_output):
        N, H_out, W_out, F = grad_output.shape
        KH, KW = self.kernel_size
        _, H_padded, W_padded, _ = self.x_padded.shape
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        grad_x_padded = np.zeros_like(self.x_padded)
        for n in range(N):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * self.stride
                    h_end = h_start + KH
                    w_start = j * self.stride
                    w_end = w_start + KW
                    for f in range(F):
                        grad_val = grad_output[n, i, j, f]
                        self.dW[..., f] += self.x_padded[n, h_start:h_end, w_start:w_end, :] * grad_val
                        self.db[f] += grad_val
                        grad_x_padded[n, h_start:h_end, w_start:w_end, :] += self.W[..., f] * grad_val
        if self.padding == 'same':
            pad_h = (KH - 1) // 2
            pad_w = (KW - 1) // 2
            if pad_h > 0 and pad_w > 0:
                grad_x = grad_x_padded[:, pad_h:-pad_h, pad_w:-pad_w, :]
            else:
                grad_x = grad_x_padded
        else:
            grad_x = grad_x_padded
        return grad_x

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

    @property
    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]


class BatchNormalization:
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        self.x_centered = None
        self.std_inv = None
        self.x = None

    def forward(self, x):
        self.x = x
        original_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(-1, original_shape[-1])
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        self.x_centered = x - mean
        self.std_inv = 1. / np.sqrt(var + self.epsilon)
        x_norm = self.x_centered * self.std_inv
        out = self.gamma * x_norm + self.beta
        # Atualiza as estatísticas de running
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        out = out.reshape(original_shape)
        return out

    def backward(self, grad_output):
        x = self.x
        original_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(-1, original_shape[-1])
            grad_output = grad_output.reshape(x.shape)
        N, D = x.shape
        x_norm = self.x_centered * self.std_inv
        dgamma = np.sum(grad_output * x_norm, axis=0, keepdims=True)
        dbeta = np.sum(grad_output, axis=0, keepdims=True)
        dx_norm = grad_output * self.gamma
        dvar = np.sum(dx_norm * self.x_centered * -0.5 * (self.std_inv**3), axis=0, keepdims=True)
        dmean = np.sum(dx_norm * -self.std_inv, axis=0, keepdims=True) + dvar * np.mean(-2. * self.x_centered, axis=0, keepdims=True)
        dx = dx_norm * self.std_inv + dvar * 2 * self.x_centered / N + dmean / N
        self.dgamma = dgamma
        self.dbeta = dbeta
        dx = dx.reshape(original_shape)
        return dx

    def update(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta

    @property
    def parameters(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


class MaxPooling2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride
        self.x = None
        self.arg_max = None

    def forward(self, x):
        self.x = x
        N, H, W, C = x.shape
        pool_h, pool_w = self.pool_size
        H_out = (H - pool_h) // self.stride + 1
        W_out = (W - pool_w) // self.stride + 1
        out = np.zeros((N, H_out, W_out, C))
        self.arg_max = np.zeros((N, H_out, W_out, C), dtype=np.int32)
        for n in range(N):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * self.stride
                    h_end = h_start + pool_h
                    w_start = j * self.stride
                    w_end = w_start + pool_w
                    window = x[n, h_start:h_end, w_start:w_end, :]
                    window_reshaped = window.reshape(-1, C)
                    out[n, i, j, :] = np.max(window_reshaped, axis=0)
                    self.arg_max[n, i, j, :] = np.argmax(window_reshaped, axis=0)
        return out

    def backward(self, grad_output):
        N, H, W, C = self.x.shape
        pool_h, pool_w = self.pool_size
        H_out = (H - pool_h) // self.stride + 1
        W_out = (W - pool_w) // self.stride + 1
        grad_input = np.zeros_like(self.x)
        for n in range(N):
            for i in range(H_out):
                for j in range(W_out):
                    for c in range(C):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        flat_index = self.arg_max[n, i, j, c]
                        idx_h = flat_index // pool_w
                        idx_w = flat_index % pool_w
                        grad_input[n, h_start+idx_h, w_start+idx_w, c] += grad_output[n, i, j, c]
        return grad_input 

class Linear:
    def __init__(self, in_features, out_features):
        # Inicializa os pesos com uma distribuição normal escalonada
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.b = np.zeros((1, out_features))
        # Variáveis para armazenar os dados necessários no backward
        self.input = None
        self.dW = None
        self.db = None

    def forward(self, x):
        # Armazena a entrada para uso no backward
        self.input = x
        # Aplica a transformação linear
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output):
        # Calcula os gradientes
        self.dW = np.dot(self.input.T, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        # Calcula o gradiente em relação à entrada
        grad_input = np.dot(grad_output, self.W.T)
        return grad_input

    def update(self, lr):
        # Atualiza os parâmetros usando descida de gradiente
        self.W -= lr * self.dW
        self.b -= lr * self.db

    @property
    def parameters(self):
        # Retorna os parâmetros e seus gradientes para uso em otimizadores externos
        return [(self.W, self.dW), (self.b, self.db)] 