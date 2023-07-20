# Attention
As of right now, due to changes that Twitter made in regards to the default page for signed out users, snscrape does not work therefore the program cannot collect tweets.

# Twitter-Sentiment
This program is used to see what the general sentiment is towards the current sitting United States president from the perspective of Twitter users. This is achieved by collecting five thousand tweets using the Python library snscrape and running those tweets through the natural language prcessing model RoBERTa. The Tweets are collecetd starting from the time of inaguration and the sentiment analysis from RoBERTa is processed to ouput what percentages of the tweets were positive, neutral, and negative.
