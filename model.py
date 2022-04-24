from transformers import pipeline
import os

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# text = """
# Python is an interpreted high-level general-purpose programming language. Its design philosophy emphasizes code readability with its use of significant indentation. Its language constructs as well as its object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.[30]

# Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly, procedural), object-oriented and functional programming. It is often described as a "batteries included" language due to its comprehensive standard library.[31]

# Guido van Rossum began working on Python in the late 1980s, as a successor to the ABC programming language, and first released it in 1991 as Python 0.9.0.[32] Python 2.0 was released in 2000 and introduced new features, such as list comprehensions and a garbage collection system using reference counting. Python 3.0 was released in 2008 and was a major revision of the language that is not completely backward-compatible. Python 2 was discontinued with version 2.7.18 in 2020.[33]

# Python consistently ranks as one of the most popular programming languages.[34][35][36][37]"""


def Text_Summarizer(text):

    # Load tokenizer
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

# Load model
    model = PegasusForConditionalGeneration.from_pretrained(
        "google/pegasus-xsum")


# Create tokens - number representation of our text
    tokens = tokenizer(text, truncation=True,
                       padding="longest", return_tensors="pt")

# Summarize
    summary = model.generate(**tokens)

# Decode summary
    return tokenizer.decode(summary[0])


def Text_Summarizer2(text, maxlen, minlen):
    summarizer = pipeline("summarization")

    # summarizer = pipeline("summarization", model="t5-base",
    #                       tokenizer="t5-base", framework="tf")

    summary_text = summarizer(
        text, max_length=maxlen, min_length=minlen, do_sample=False)[0]['summary_text']

    return summary_text


def text_cleaner(text, num):
    newString = text.lower()                         # Convert to lower case
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"', '', newString)
    newString = ' '.join(
        [contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
    newString = re.sub(r"'s\b", "", newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    newString = re.sub('[m]{2,}', 'mm', newString)
    if(num == 0):
        tokens = [w for w in newString.split() if not w in stop_words]
    else:
        tokens = newString.split()
    long_words = []
    for i in tokens:
        if len(i) > 1:  # removing short word
            long_words.append(i)
    return (" ".join(long_words)).strip()
