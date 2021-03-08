from pathos.multiprocessing import ProcessingPool as Pool

def clean_text(sentences: list, eos: str):
    def clean(text):
        text = text.replace("@@ ", "")
        return "".join(text.partition(eos)[0]).strip().strip("@@")

    with Pool() as p:
        sentences = p.map(clean, sentences)

    return sentences
