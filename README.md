# ezllm

ezllm é uma biblioteca super fácil e otimizada para treinamento de redes neurais usando NumPy. Ela permite criar, treinar e avaliar modelos com poucas linhas de código, sendo útil tanto para tarefas de regressão quanto para problemas de classificação e processamento de linguagem natural.

## Instalação

Para instalar a biblioteca localmente, execute:
```bash
pip install -e .
```

## Exemplo de uso

A seguir, um exemplo detalhado de como usar o ezllm para resolver um problema de regressão (y = 2 * x + 1). Este exemplo ilustra a criação do modelo, a configuração do otimizador, o treinamento e a visualização dos resultados. Além disso, veja os comentários ao final que explicam como a lib pode ser aplicada em outras situações, como conversas simples e jogos (veja os testes: test_talk.py, test_rps.py e test_ezllm.py).

```python
import numpy as np
import matplotlib.pyplot as plt
from ezllm import Model, Dense, ReLU, mse_loss, SGD

# Gerar dataset para regressão: y = 2 * x + 1
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = 2 * X + 1

# Definindo o modelo neural:
# - Primeira camada Dense: de 1 entrada para 10 neurônios.
# - Função de ativação ReLU.
# - Segunda camada Dense: de 10 neurônios para 1 saída.
model = Model([
    Dense(1, 10),
    ReLU(),
    Dense(10, 1)
])

# Configurando o otimizador SGD com uma taxa de aprendizado de 0.01
optimizer = SGD(model.parameters, lr=0.01)

# Treinamento do modelo por 100 épocas
print("Iniciando o treinamento do modelo de regressão...")
model.fit(X, y, epochs=100, optimizer=optimizer)
print("Treinamento concluído!")

# Realizando predições com o modelo treinado
y_pred = model.predict(X)

# Visualizando os resultados: dados reais vs. predições do modelo
plt.scatter(X, y, label='Dados reais')
plt.plot(X, y_pred, label='Predição do modelo', color='red')
plt.title("Exemplo de Regressão com ezllm")
plt.legend()
plt.show()

# Observações:
#  - Para criar modelos de conversação, consulte test_talk.py,
#    onde o Tokenizer é combinado com uma rede neural simples para aprender respostas.
#  - Para problemas de classificação, como o jogo Pedra, Papel, Tesoura,
#    veja test_rps.py, que demonstra o uso de vetorização one-hot e a função cross_entropy_loss.
#  - O arquivo test_ezllm.py contém exemplos adicionais, como o uso do mse_loss em tarefas de regressão.
```