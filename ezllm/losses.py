import numpy as np

def mse_loss(y_pred, y_true):
    """
    Calcula o erro quadrático médio (MSE) e seu gradiente.
    """
    loss = np.mean((y_pred - y_true) ** 2)
    grad = 2 * (y_pred - y_true) / y_pred.shape[0]
    return loss, grad

def softmax(x):
    # Estabilidade numérica: subtrai o máximo de cada linha
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    """
    Calcula a loss de Cross Entropy para classificadores multi-classe.
    Assume que y_true está no formato one-hot.
    """
    p = softmax(y_pred)
    loss = -np.mean(np.sum(y_true * np.log(p + 1e-7), axis=1))
    grad = (p - y_true) / y_pred.shape[0]
    return loss, grad 