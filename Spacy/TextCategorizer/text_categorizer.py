import spacy
import en_core_web_md


if __name__ == "__main__":
    nlp = en_core_web_md.load()
    doc = nlp(u'hello spacy!')
    print(doc.text)
