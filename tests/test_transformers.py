import numpy as np
from ezllm.transformers import Transformer
from ezllm.losses import cross_entropy_loss
from ezllm.transformers import TransformerTokenizer
from ezllm.transformers import ensure_3d
from ezllm.transformers import create_onehot

def test_transformer_text_completion():
    # Define um corpus simples para completamento de texto
    corpus = ["eu", "gosto", "de", "pizza", "e", "eu", "tambem", "gosto", "de", "sorvete", "<eos>"] * 20
    
    # Usa apenas o TransformerTokenizer
    tokenizer = TransformerTokenizer(lower=True, sep=" ")
    tokenizer.fit(corpus)
    vocab_size = tokenizer.vocab_size()
    
    # Cria sequências de input e target
    input_text = corpus[:-1]
    target_text = corpus[1:]
    
    # Codifica os tokens usando o TransformerTokenizer
    X = tokenizer.encode_onehot(input_text)
    y = tokenizer.encode_onehot(target_text)
    
    # Adiciona dimensão de batch
    X = np.expand_dims(X, axis=0)
    y = np.expand_dims(y, axis=0)

    # Cria e treina o modelo Transformer
    model = Transformer(
        num_layers=2,    # 2 camadas/blocos Transformer
        d_model=16,      # Vetores de embedding de tamanho 16
        num_heads=2,     # 2 cabeças de atenção
        d_ff=32,         # Rede feed-forward com camada interna de tamanho 32
        vocab_size=vocab_size   # Vocabulário de tamanho 100
    )
    
    epochs = 200
    lr = 0.05
    model.fit(X, y, epochs=epochs, lr=lr, loss_fn=cross_entropy_loss, verbose=True)

    # Testa a geração de texto utilizando o método generate do modelo.
    prompt = "eu"
    print("\nTestando geração de texto:")
    for temp in [0.5, 1.0, 1.5]:
        generated_text = model.generate(prompt, tokenizer, max_tokens=4, temperature=temp)
        print(f"Temperatura {temp}: {generated_text}")
        assert len(generated_text.split()) > 1, f"Falha na geração com temperatura {temp}"
    model.save("model.pkl")

    loaded_model = Transformer.load("model.pkl")
    print(loaded_model.generate(prompt, tokenizer, max_tokens=4, temperature=0.5))

if __name__ == "__main__":
    test_transformer_text_completion()
    print("Teste de transformers e completador de texto passou!") 