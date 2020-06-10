from textblob.classifiers import NaiveBayesClassifier


def trainer():
    """Trainer function for Naive Bayers classifier"""

    train = [
        ("i am good", "pos"),
        ("fine thanks", "pos"),
        ("doing ok", "pos"),
        ("thank you", "pos"),
        ("i am crappy", "neg"),
        ("omg", "neg"),
        ("no thanks", "pos"),
        ("do you have information on the SDK", "pos"),
        ("not good", "neg"),
        ("you are awesome", "pos"),
        ("i don't know", "neg"),
        ("forget you", "neg"),
        ("it didn't answer my question", "neg")
    ]

    return NaiveBayesClassifier(train)

if __name__ == "__main__":
    user_input = "I find this helpful"
    classy = trainer().prob_classify(user_input)
    print()
    print(f'String:  {user_input} ')
    print(f'---------{len(user_input)* "-"}+')
    print(f'Negative probability: {classy.prob("neg")}')
    print(f'Positive probability: {classy.prob("pos")}')