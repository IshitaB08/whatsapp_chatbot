from textblob import TextBlob
import twily_classifier as cl
import stop_words as stop_words
import json

with open('twilybot.json', 'r') as f:
    array = json.load(f)

CONVERSATION = array["conversations"]

BOT_NAME = 'Cheems'
STOP_WORDS = stop_words.sw_list
neg_distribution = []

def sentiment(u_input):
    blob_it = cl.trainer().prob_classify(u_input)
    npd = round(blob_it.prob("neg"), 2)
    neg_distribution.append(npd)
    return npd

def simplebot(user_input):
    user_blob = TextBlob(user_input)
    lower_input = user_blob.lower()
    token_input = lower_input.words
    filtered_input = [w for w in token_input if w not in STOP_WORDS]
    response_set = set()
    for con_list in CONVERSATION:
        for sentence in con_list:
            sentence_split = sentence.split()
            if set(filtered_input).intersection(sentence_split):
                response_set.update(con_list)
    if not response_set:
        return "I'm sorry, ask again please!"
    else:
        return max(response_set, key=len)

def escalation(user_input):
   # Takes an argument, user_input, in form of a string ...

    live_rep = f'We apologize {BOT_NAME} is unable to assist you, we are getting a live representative for you, please stay with us'

    sentiment(user_input)
    list_len = len(neg_distribution)
    bot_response = simplebot(user_input)
    if list_len > 3:
        last_3 = neg_distribution[-3:]
        if last_3[0] > .40 and last_3[0] <= last_3[1] <= last_3[2]:
            return live_rep
        else:
            return bot_response
    else:
        return bot_response

# if __name__ == '__main__':
#     while True:
#         try:
#             user_input = input('You: ')
#             print(escalation(user_input))
#             print(neg_distribution)
#         except (KeyboardInterrupt, EOFError, SystemExit):
#             break