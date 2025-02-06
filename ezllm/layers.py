import numpy as np

class Dense:
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
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output):
        # Calcula o gradiente dos pesos e do bias
        self.dW = np.dot(self.input.T, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        # Retorna o gradiente em relação à entrada para o fluxo de retropropagação
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