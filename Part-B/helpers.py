import re
import math

EOS = "EOS"
SOS = "SOS"
EOS_token = 0  # A special token representing the end of a sequence
SOS_token = 1  # A special token representing the end of a sequence


class Vocab:
    def __init__(self, name):
        self.name = name  # The name of the vocabulary
        self._word2token = {EOS: EOS_token, SOS: SOS_token}  # Map words to token index
        self._word2count = {
            EOS: 0,
            SOS: 0,
        }  # Track how many times a word occurs in a corpus
        self._token2word = {
            EOS_token: EOS,
            SOS_token: SOS,
        }  # Map token indexs back into words
        self._n_words = (
            2  # Count SOS and EOS                # Number of unique words in the corpus
        )

    # Get a list of all words
    def get_words(self):
        return list(self._word2count.keys())

    # Get the number of words
    def num_words(self):
        return self._n_words

    # Convert a word into a token index
    def word2token(self, word):
        return self._word2token[word]

    # Convert a token into a word
    def token2word(self, word):
        return self._token2word[word]

    # Get the number of times a word occurs
    def word2count(self, word):
        return self._word2count[word]

    # Add all the words in a sentence to the vocabulary
    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    # Add a single word to the vocabulary
    def add_word(self, word):
        if word not in self._word2token:
            self._word2token[word] = self._n_words
            self._word2count[word] = 1
            self._token2word[self._n_words] = word
            self._n_words += 1
        else:
            self._word2count[word] += 1


def normalize_string(s):
    """
    Separate punctuation into separate tokens.
    Remove anything that is not a letter, number, or punctuation.
    """
    s = s.strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z0-9 ]+", r"", s)
    return s


def make_vocab(docs, vocab_name=""):
    """
    Pass in observations from the game, record each unique word.
    """
    vocab = Vocab(vocab_name)
    for s in docs:
        words = normalize_string(s).split()
        for w in words:
            vocab.add_word(w)

    return vocab


def insert_eos_sos(observations):
    observations = re.sub(r"\n+", " ", observations).strip().lower()
    sentences = re.split(r"([.!?])", observations)

    merged_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip() + sentences[i + 1]
        merged_sentences.append(sentence)

    wrapped_sentences = [
        f"{SOS} {sentence} {EOS}" for sentence in merged_sentences if sentence.strip()
    ]

    return " ".join(wrapped_sentences)


def perplexity1(sequence, unigram_lm):
    # Start with 0 (adding, not multiplying)
    p = 0.0
    # Add log probability of each token
    for n in range(0, len(sequence)):
        if unigram_lm[sequence[n]] == 0:
            p = p + (-10)
        else:
            p = p + math.log(unigram_lm[sequence[n]])
    # Average, flip to positive scores
    return -p / len(sequence)


def perplexity2(sequence, unigram_lm, bigram_lm):
    # Start with log probability of the first token, which has no predecessors
    p = 0.0
    if unigram_lm[sequence[0]] == 0:
        p = p + (-10)
    else:
        p = math.log(unigram_lm[sequence[0]])
    # Add log probability for each bigram
    for n in range(1, len(sequence)):
        if bigram_lm[sequence[n - 1]][sequence[n]] == 0:
            p = p + (-10)
        else:
            p = p + math.log(bigram_lm[sequence[n - 1]][sequence[n]])
    # Average, flip to positive scores
    return -p / len(sequence)


def perplexity3(sequence, unigram_lm, bigram_lm, trigram_lm):
    # Start with log probability of the firs token plus log probability of first bigram
    if unigram_lm[sequence[0]] == 0:
        p = -10
    else:
        p = math.log(unigram_lm[sequence[0]])

    if bigram_lm[sequence[0]][sequence[1]] == 0:
        p += -10
    else:
        p += math.log(bigram_lm[sequence[0]][sequence[1]])
    # Add log probability for each trigram
    for n in range(2, len(sequence)):
        if trigram_lm[sequence[n - 2]][sequence[n - 1]][sequence[n]] == 0:
            p = p + (-10)
        else:
            p = p + math.log(trigram_lm[sequence[n - 2]][sequence[n - 1]][sequence[n]])
    # Average, flip to positive scores
    return -p / len(sequence)
