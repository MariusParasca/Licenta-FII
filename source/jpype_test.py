import jpype


classpath = r".\3rd party java\stanford-parser.jar"
jpype.startJVM(r"C:\Program Files\Java\jdk1.8.0_161\jre\bin\server\jvm.dll", "-Djava.class.path=%s" % classpath)
jpype.java.lang.System.out.println("hello world")
nlp = jpype.JPackage("edu").stanford.nlp
tagger = nlp.tagger.maxent.MaxentTagger(".\stanford-postagger-2018-10-16\models\english-left3words-distsim.tagger")
tokenizeText = nlp.tagger.maxent.MaxentTagger.tokenizeText
text = jpype.java.io.StringReader("Karma of humans is AI")
sentences = tokenizeText(text)
for sentence in sentences:
    tSentence = tagger.tagSentence(sentence)
    # result = " ".join(list(tSentence))
    string = str(tSentence.toString())
    print(string.replace("[", "").replace("]", "").replace(", ", " "))

jpype.shutdownJVM()

