import numpy as np
from ezllm.transformers import Transformer
from ezllm.losses import cross_entropy_loss
from ezllm.transformers import TransformerTokenizer
from ezllm.transformers import ensure_3d
from ezllm.transformers import create_onehot

def test_transformer_text_completion():
    # Define um corpus simples para completamento de texto
    corpus = ["eu", "gosto", "de", "pizza", "e", "eu", "tambem", "gosto", "de", "sorvete", "<eos>"]
    
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

    # Função auxiliar para geração de texto
    def generate_text(model, tokenizer, prompt, max_tokens=5, temperature=1.0):
        generated = [prompt]
        input_token = tokenizer.encode(prompt)
        
        # Usa a função ensure_3d para garantir o formato correto
        input_token = ensure_3d(input_token)
        
        for _ in range(max_tokens):
            # Usa o método predict_token do modelo
            next_token_idx = model.predict_token(input_token, temperature=temperature)
            next_token = tokenizer.idx2word[next_token_idx[0]]
            generated.append(next_token)
            
            # Para se encontrar o token de fim de sequência
            if next_token == "<eos>":
                break
                
            # Cria vetor one-hot do token gerado usando a nova função
            next_onehot = create_onehot(next_token_idx[0], tokenizer.vocab_size())
            
            # Garante que next_onehot tenha 3 dimensões
            next_onehot = ensure_3d(next_onehot)
            
            input_token = np.concatenate([input_token, next_onehot], axis=1)
        
        return " ".join(generated)

    # Teste de geração de texto com diferentes temperaturas
    prompt = "eu"
    print("\nTestando geração de texto:")
    for temp in [0.5, 1.0, 1.5]:
        generated_text = generate_text(model, tokenizer, prompt, max_tokens=4, temperature=temp)
        print(f"Temperatura {temp}: {generated_text}")
        assert len(generated_text.split()) > 1, f"Falha na geração com temperatura {temp}"

if __name__ == "__main__":
    test_transformer_text_completion()
    print("Teste de transformers e completador de texto passou!") 