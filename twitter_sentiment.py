import snscrape.modules.twitter as sntwitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# searching from the start of Biden's presidency
query = "Biden lang:en since:2021-01-20"
limitOfTweets = 5000
tweets = []

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) == limitOfTweets:
        break
    else:
        tweets.append(tweet.content)
    
# preprocess the tweets in order to make them usable for model
tweetWords = []

for tweet in tweets:
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweetWords.append(word)
    tweet = " ".join(tweetWords)    
  
# loading in model and tokenizer  
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

# sentiment analysis
labels = ["Negative", "Neutral", "Positive"]
avgScores = [0]*3

for tweet in tweets:
    encodedTweet = tokenizer(tweet, return_tensors='pt')
    output = model(**encodedTweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    for i in range(3):
        avgScores[i] += scores[i]
        
print("\nThe total sentiment of the current president from twitter users are as follows.")
print("The closer the numbers are to 1, the higher that sentiment is.\n")
for i in range(3):
    avgScores[i] /= len(tweets)
    l = labels[i]
    s = avgScores[i]
    print(l, s)
