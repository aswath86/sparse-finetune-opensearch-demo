#!/usr/bin/env python3
"""
Build IDF weights from PubMed abstracts using PubMedBERT's tokenizer.

Downloads the pubmed_qa unlabeled split (~61K abstracts) from HuggingFace,
tokenizes with PubMedBERT, computes IDF = log(N / df), then optionally
zeros out stopwords to produce a clean version.

Usage:
    python build_idf.py
    python build_idf.py --output idf_pubmedbert.json
    python build_idf.py --output idf_pubmedbert_clean.json --zero-stopwords

Outputs:
    idf_pubmedbert.json        — raw IDF (all tokens)
    idf_pubmedbert_clean.json  — stopwords zeroed (use this for training)
"""
import argparse, json, math
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re",
    "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
    "won", "wouldn",
    # informal words rare in PubMed that get artificially high IDF
    "something", "anything", "everything", "nothing", "someone", "anyone",
    "everyone", "nobody", "gonna", "wanna", "gotta", "kinda", "sorta",
    "ok", "okay", "yeah", "yes", "no", "hey", "hi", "hello", "bye",
    "please", "thanks", "thank", "sorry", "excuse", "wow", "oh", "ah",
    "um", "uh", "hmm", "huh", "oops", "yep", "nope", "sure", "maybe",
    "probably", "definitely", "absolutely", "exactly", "really", "actually",
    "basically", "literally", "seriously", "obviously", "apparently",
    "pretty", "quite", "rather", "enough", "almost", "already", "also",
    "always", "never", "sometimes", "often", "usually", "still", "yet",
    "even", "just", "already", "anyway", "besides", "however", "though",
    "although", "unless", "whether", "since", "therefore", "thus",
    "hence", "meanwhile", "otherwise", "instead", "regardless",
    "thing", "things", "stuff", "lot", "lots", "bit", "kind", "type",
    "way", "ways", "time", "times", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "today", "tomorrow", "yesterday",
    "morning", "night", "tonight", "ago", "later", "soon", "early", "late",
    "good", "bad", "great", "nice", "fine", "well", "better", "best",
    "worse", "worst", "big", "small", "little", "much", "many", "few",
    "long", "short", "old", "new", "young", "high", "low", "right", "wrong",
    "hard", "easy", "fast", "slow", "hot", "cold", "warm", "cool",
    "get", "got", "getting", "go", "going", "went", "gone", "come",
    "came", "coming", "make", "made", "making", "take", "took", "taken",
    "give", "gave", "given", "keep", "kept", "let", "say", "said",
    "tell", "told", "ask", "asked", "think", "thought", "know", "knew",
    "see", "saw", "seen", "look", "looked", "want", "wanted", "need",
    "needed", "try", "tried", "use", "used", "find", "found", "put",
    "mean", "meant", "feel", "felt", "leave", "left", "call", "called",
    "like", "liked", "work", "worked", "seem", "seemed", "help", "helped",
    "show", "showed", "turn", "turned", "move", "moved", "live", "lived",
    "believe", "bring", "happen", "write", "provide", "sit", "stand",
    "lose", "pay", "meet", "play", "run", "hold", "learn", "change",
    "lead", "understand", "watch", "follow", "stop", "create", "speak",
    "read", "allow", "add", "spend", "grow", "open", "walk", "win",
    "teach", "offer", "remember", "consider", "appear", "buy", "wait",
    "serve", "die", "send", "expect", "build", "stay", "fall", "cut",
    "reach", "kill", "remain",
    "people", "person", "man", "woman", "child", "children", "kid", "kids",
    "boy", "girl", "baby", "family", "friend", "friends", "home", "house",
    "school", "place", "world", "country", "city", "state", "part",
    "hand", "head", "eye", "eyes", "face", "back", "side", "end",
    "number", "point", "fact", "case", "question", "problem", "idea",
    "story", "word", "words", "name", "group", "company", "system",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="idf_pubmedbert.json")
    p.add_argument("--zero-stopwords", action="store_true",
                   help="Zero out stopword IDF values (recommended for training)")
    args = p.parse_args()

    print(f"Loading tokenizer: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)

    print("Loading pubmed_qa unlabeled split...")
    ds = load_dataset("pubmed_qa", "pqa_unlabeled", split="train")

    print(f"Computing IDF from {len(ds)} abstracts...")
    df_count = Counter()
    N = 0
    for row in ds:
        text = " ".join(row["context"]["contexts"])
        ids = set(tok.encode(text, add_special_tokens=False))
        for i in ids:
            df_count[i] += 1
        N += 1

    idf = {}
    for token_id, count in df_count.items():
        token = tok.decode([token_id]).strip()
        if token:
            idf[token] = round(math.log(N / count), 4)

    zeroed = 0
    if args.zero_stopwords:
        for token in list(idf.keys()):
            if token.lower() in STOPWORDS:
                idf[token] = 0
                zeroed += 1

    with open(args.output, "w") as f:
        json.dump(idf, f)

    print(f"Wrote {len(idf)} tokens to {args.output} ({N} docs)")
    if args.zero_stopwords:
        print(f"Zeroed {zeroed} stopword entries")


if __name__ == "__main__":
    main()
