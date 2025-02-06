class SGD:
    def __init__(self, parameters, lr=0.01):
        """
        parameters: uma lista de tuplas (param, grad) para atualização
        lr: taxa de aprendizado
        """
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param, grad in self.parameters:
            param -= self.lr * grad

    def zero_grad(self):
        # Zera os gradientes
        for _, grad in self.parameters:
            grad[...] = 0  # Reseta os valores para zero 