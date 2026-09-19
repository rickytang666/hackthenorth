import hashlib
import json
import random
import urllib.request

def emit(kind, **fields):
    print(json.dumps(dict(kind=kind, **fields)), flush=True)

SOURCES = {
    "wiki": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
    "python": "https://raw.githubusercontent.com/python/cpython/3.12/Lib/heapq.py",
    "literature": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}
CASES = [(1,512,32), (2,512,64), (4,2048,32), (8,512,64), (16,512,128)]
CONFIGS = [(3,2,2), (3,4,4), (4,4,2)]
TRIALS = 5


def corpora(tokenizer):
    streams = {}
    for name, url in SOURCES.items():
        raw = urllib.request.urlopen(url, timeout=60).read()
        streams[name] = tokenizer.encode(raw.decode("utf-8"), add_special_tokens=False)
        emit("corpus", name=name, url=url, sha256=hashlib.sha256(raw).hexdigest(),
             tokens=len(streams[name]))
    return streams


def prompts(tokenizer, streams, shape, trial):
    batch, length, _ = shape
    inputs, kinds = [], []
    instructions = {
        "wiki": "Summarize the passage and explain its main points in your own words.",
        "python": "Explain the code and discuss its time complexity and edge cases.",
        "literature": "Explain the characters' motivations and the conflict in this passage.",
    }
    for row in range(batch):
        rng = random.Random(7719 + batch*100000 + length*11 + trial*103 + row*1009)
        kind = (trial+row) % 6
        name = list(SOURCES)[kind % 3]
        prefix, suffix = [], []
        if kind >= 3:
            prefix = tokenizer.encode("<|im_start|>user\nPassage:\n", add_special_tokens=False)
            suffix = tokenizer.encode("\n\n"+instructions[name]+"<|im_end|>\n<|im_start|>assistant\n",
                                      add_special_tokens=False)
        needed = length-len(prefix)-len(suffix)
        start = rng.randrange(len(streams[name])-needed)
        ids = prefix+streams[name][start:start+needed]+suffix
        assert len(ids) == length
        inputs.append(ids)
        kinds.append(name+("_instruction" if kind >= 3 else "_continuation"))
    return inputs, kinds

