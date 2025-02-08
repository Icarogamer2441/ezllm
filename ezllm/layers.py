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
        if self.num_attention_layers > 0 and self.input.ndim == 3 and grad_output.ndim == 3:
            batch_size, seq_len, output_dim = grad_output.shape
            input_dim = self.W.shape[0]
            # Reshape para (batch_size * seq_len, output_dim)
            grad_output_reshaped = grad_output.reshape(-1, output_dim)
            input_reshaped = self.input.reshape(-1, input_dim)
            # Calcula o gradiente dos pesos e do bias
            self.dW = np.dot(input_reshaped.T, grad_output_reshaped)
            self.db = np.sum(grad_output_reshaped, axis=0, keepdims=True)
            # Retorna o gradiente em relação à entrada
            grad_input_reshaped = np.dot(grad_output_reshaped, self.W.T)
            grad_input = grad_input_reshaped.reshape(batch_size, seq_len, input_dim)
        else:
            # Se o grad_output vier com uma dimensão extra (por exemplo, shape (batch_size, 1, out_features)),
            # squeeze para transformar em 2D.
            if grad_output.ndim == 3 and grad_output.shape[1] == 1:
                grad_output = grad_output.squeeze(1)
            # Para entradas 2D, calcula os gradientes diretamente
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