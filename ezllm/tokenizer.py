import numpy as np

class Tokenizer:
    def __init__(self, lower=True, sep=" "):
        self.lower = lower
        self.sep = sep
        self.word2idx = {}
        self.idx2word = {}

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