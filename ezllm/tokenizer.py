import numpy as np
from PIL import Image
from ezllm.layers import Dense

class Tokenizer:
    def __init__(self, lower=True, sep=" "):
        self.lower = lower
        self.sep = sep
        self.word2idx = {}
        self.idx2word = {}
        self.color_mode = False  # Novo atributo para modo de cor

    def fit(self, texts):
        """
        texts: lista de strings
        Cria o vocabulário com base nos tokens encontrados.
        """
        tokens = set()
        for text in texts:
            if self.lower:
                text = text.lower()
            tokens.update(text.split(self.sep))
        tokens = sorted(tokens)
        self.word2idx = {word: idx for idx, word in enumerate(tokens)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, text):
        """
        Converte uma string em uma lista de inteiros (tokens).
        """
        if self.lower:
            text = text.lower()
        return [self.word2idx[word] for word in text.split(self.sep) if word in self.word2idx]

    def decode(self, tokens):
        """
        Converte uma lista de tokens (inteiros) para a string original.
        """
        return self.sep.join([self.idx2word[token] for token in tokens if token in self.idx2word])

    def vocab_size(self):
        """
        Retorna o tamanho do vocabulário.
        """
        return len(self.word2idx)

    def encode_onehot(self, texts):
        """
        Converte um texto ou uma lista de textos em codificação one-hot.
        Se 'texts' for uma string, a função primeiro tokeniza o texto e retorna um
        array 2D com cada token codificado em one-hot (ou um único vetor caso haja
        apenas um token).

        Se for uma lista de strings, assume que cada string contém apenas um token
        e retorna um array 2D onde cada linha é a codificação one-hot da respectiva string.
        """
        if isinstance(texts, str):
            tokens = self.encode(texts)
            onehots = []
            for token in tokens:
                vec = np.zeros(self.vocab_size())
                vec[token] = 1
                onehots.append(vec)
            return np.array(onehots) if len(onehots) > 1 else onehots[0]
        elif isinstance(texts, list):
            onehots = []
            for t in texts:
                tokens = self.encode(t)
                vec = np.zeros(self.vocab_size())
                if tokens:
                    vec[tokens[0]] = 1
                onehots.append(vec)
            return np.array(onehots)

    def decode_onehot(self, onehots):
        """
        Converte uma ou mais codificações one-hot de volta para string(s).
        Se 'onehots' é um array 1D, retorna a string correspondente.
        Se for um array 2D, retorna uma lista de strings.
        """
        onehots = np.array(onehots)
        if onehots.ndim == 1:
            idx = int(np.argmax(onehots))
            return self.idx2word[idx]
        elif onehots.ndim == 2:
            words = []
            for vec in onehots:
                idx = int(np.argmax(vec))
                words.append(self.idx2word[idx])
            return words

    def predict(self, logits):
        """
        Converte as predições (logits ou one-hot) do modelo em palavra(s) correspondentes.
        Se `logits` for um array 1D, retorna a string correspondente.
        Se for um array 2D com apenas uma amostra (shape[0]==1), retorna a string correspondente.
        Se for um array 2D com múltiplas amostras, retorna uma lista de strings.
        """
        logits = np.array(logits)
        if logits.ndim == 1:
            return self.idx2word[int(np.argmax(logits))]
        elif logits.ndim == 2:
            if logits.shape[0] == 1:
                return self.idx2word[int(np.argmax(logits))]
            else:
                return [self.idx2word[i] for i in np.argmax(logits, axis=1)]

    def fit_image(self, img):
        """Processa uma imagem PIL para criar vocabulário de cores"""
        self.color_mode = True
        pixels = list(img.getdata())
        unique_colors = set(pixels)
        self.word2idx = {color: idx for idx, color in enumerate(unique_colors)}
        self.idx2word = {idx: color for color, idx in self.word2idx.items()}

    def encode_image_onehot(self, img):
        """Converte imagem para one-hot encoding"""
        pixels = list(img.getdata())
        onehots = []
        for color in pixels:
            vec = np.zeros(self.vocab_size())
            vec[self.word2idx[color]] = 1
            onehots.append(vec)
        return np.array(onehots)

    def save_image(self, onehots, filepath, original_size):
        """Decodifica one-hot para imagem e salva"""
        pixels = []
        for vec in onehots:
            idx = np.argmax(vec)
            pixels.append(self.idx2word[idx])
        
        img = Image.new('RGB', original_size)
        img.putdata(pixels)
        img.save(filepath)
        print(f"Imagem salva em {filepath}")

    def adjust_model(self, model, prev_vocab_sizes=None):
        """
        Adjusts the model's input and output layers to accommodate new vocabulary size.
        This is used before fit_another to expand the model's capacity for new data.
        
        Args:
            model: The model to adjust
            prev_vocab_sizes: List of previous vocabulary sizes to preserve weights for
        """
        # Get the new vocabulary size
        new_vocab_size = self.vocab_size()
        
        # If no previous vocab sizes provided, use the current layer sizes
        if prev_vocab_sizes is None:
            prev_vocab_sizes = [model.layers[0].W.shape[0]]
        
        # Get the first and last layers
        first_layer = model.layers[0]
        last_layer = model.layers[-1]
        
        # Adjust input layer (first layer)
        if isinstance(first_layer, Dense):
            # Create new weights matrix with additional columns for new vocabulary
            new_input_weights = np.random.randn(new_vocab_size, first_layer.W.shape[1]) * np.sqrt(2. / new_vocab_size)
            
            # Copy existing weights for each previous vocabulary size
            start_idx = 0
            for vocab_size in prev_vocab_sizes:
                end_idx = start_idx + vocab_size
                if end_idx > new_vocab_size:
                    end_idx = new_vocab_size
                new_input_weights[start_idx:end_idx, :] = first_layer.W[start_idx:end_idx, :]
                start_idx = end_idx
            
            first_layer.W = new_input_weights
            first_layer.input = None  # Reset input cache
            
        # Adjust output layer (last layer)
        if isinstance(last_layer, Dense):
            # Create new weights matrix with additional rows for new vocabulary
            new_output_weights = np.random.randn(last_layer.W.shape[0], new_vocab_size) * np.sqrt(2. / last_layer.W.shape[0])
            
            # Copy existing weights for each previous vocabulary size
            start_idx = 0
            for vocab_size in prev_vocab_sizes:
                end_idx = start_idx + vocab_size
                if end_idx > new_vocab_size:
                    end_idx = new_vocab_size
                new_output_weights[:, start_idx:end_idx] = last_layer.W[:, start_idx:end_idx]
                start_idx = end_idx
            
            last_layer.W = new_output_weights
            
            # Adjust bias
            new_bias = np.zeros((1, new_vocab_size))
            start_idx = 0
            for vocab_size in prev_vocab_sizes:
                end_idx = start_idx + vocab_size
                if end_idx > new_vocab_size:
                    end_idx = new_vocab_size
                new_bias[:, start_idx:end_idx] = last_layer.b[:, start_idx:end_idx]
                start_idx = end_idx
                
            last_layer.b = new_bias
            last_layer.input = None  # Reset input cache 