class SimpleVocab:
    def __init__(self):
        self.token_to_ids = {
            "[PAD]": 0,
            "[MASK]": 1,
            "[NO_USE]": 2
        }
        self.id_to_tokens = {v: k for k, v in self.token_to_ids.items()}
        self.vocab_words = list(self.token_to_ids.keys())  # ✅ 추가

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_ids[t] for t in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_tokens[i] for i in ids]

    def get_pad_id(self):
        return self.token_to_ids["[PAD]"]

    def get_mask_id(self):
        return self.token_to_ids["[MASK]"]
