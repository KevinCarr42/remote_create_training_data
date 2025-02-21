import json
from pathlib import Path


class LanguageClassifier:
    def __init__(self, wordlist_path=None, max_incorrect_words=0, min_words_in_language=2):
        default_path = Path(__file__).parent / "wordlists.json"
        wordlist_path = wordlist_path or default_path

        with open(wordlist_path, "r", encoding="utf-8") as f:
            wordlists = json.load(f)

        self.french_words = set(wordlists["fr"])
        self.english_words = set(wordlists["en"])
        self.max_incorrect_words = max_incorrect_words
        self.min_words_in_language = min_words_in_language

    def classify(self, sentence):
        words = sentence.lower().split()
        en_count = sum(1 for word in words if word in self.english_words)
        fr_count = sum(1 for word in words if word in self.french_words)

        if en_count > fr_count and en_count >= self.min_words_in_language and fr_count <= self.max_incorrect_words:
            return 'en'
        if fr_count > en_count and fr_count >= self.min_words_in_language and en_count <= self.max_incorrect_words:
            return 'fr'
        if (fr_count >= self.max_incorrect_words and en_count >= self.max_incorrect_words and
                fr_count + en_count >= self.min_words_in_language):
            return 'mixed'

        return 'unknown'
