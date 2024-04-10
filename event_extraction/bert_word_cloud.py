import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv('predicted_sentences_bert_final.csv')
# Combine all the sentences into one large string
text = ' '.join(df['cleaned_text'].astype(str))

# Generate a word cloud image
wordcloud = WordCloud(background_color='white').generate(text)

# Display the generated image
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('bert_wordcloud.png')
plt.show()