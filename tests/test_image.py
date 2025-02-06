import numpy as np
from PIL import Image
from ezllm.core import Model
from ezllm.layers import Dense, ReLU
from ezllm.losses import cross_entropy_loss
from ezllm.tokenizer import Tokenizer
import os
import argparse

def test_image_generation(input_path="input.png", output_path=None):
    # Carrega a imagem do arquivo especificado
    try:
        img = Image.open(input_path)
    except FileNotFoundError:
        # Fallback para imagem teste se não encontrar o arquivo
        img = Image.new('RGB', (2, 2), color='red')
    
    # Gera nome do arquivo de saída se não for especificado
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_generated"  # Sem extensão

    # Cria e ajusta o tokenizador para cores
    tokenizer = Tokenizer(lower=False, sep=None)  # sep=None para modo imagem
    tokenizer.fit_image(img)
    
    # Codifica a imagem em one-hot (formato achatado)
    X = tokenizer.encode_image_onehot(img)
    y = X  # Modelo auto-regressivo (aprende a reproduzir a entrada)

    # Cria modelo neural simples
    model = Model([
        Dense(X.shape[1], 64),
        ReLU(),
        Dense(64, 32),
        ReLU(),
        Dense(32, X.shape[1])
    ])

    # Treina o modelo com a nova opção
    print("Treinando modelo para geração de imagem...")
    model.fit(X, y, epochs=1000, lr=0.1,
             loss_fn=cross_entropy_loss, verbose=0,
             progress_bar=True, test_train=True,
             output_test_train=True, output_interval=10,  # Gera a cada 10 épocas
             tokenizer=tokenizer, output_path=output_path, img_size=img.size)

    # Gera a imagem a partir do modelo
    generated = model.predict(X)
    tokenizer.save_image(generated, output_path, img.size)

    print(f"Imagem gerada salva como: {output_path}")

    # Verificações básicas
    output_img = Image.open(output_path)
    assert output_img.size == img.size, "Tamanho da imagem gerada incorreto"
    assert os.path.exists(output_path), "Arquivo de imagem não foi gerado"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gerador de imagens com IA')
    parser.add_argument('-i', '--input', type=str, default="input.png",
                       help='Caminho da imagem de entrada (padrão: input.png)')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Caminho da imagem de saída (opcional)')
    
    args = parser.parse_args()
    test_image_generation(input_path=args.input, output_path=args.output)
    print("Teste de geração de imagem passou!")
    # usage: python test_image.py -i input.png -o output.png