import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

capital_file = 'Marx_Capital_1.pdf'
capital = open(capital_file, 'rb')
capitalReader = PyPDF2.PdfReader(capital)

# The actual text starts on page 124
first_page = 123
last_page = 1083
page_text = []
for pageNum in range(first_page, last_page):
    page_text.append(capitalReader.pages[pageNum].extract_text())

vectorizer = CountVectorizer(stop_words='english', min_df=0.01, max_df=0.95)
X = vectorizer.fit_transform(page_text)
features = vectorizer.get_feature_names_out()
idx_word = 'religion'
word_idx = np.where(features == idx_word)[0][0]

df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(df.iloc[0:5, word_idx])
# index = ['453-454', 752, 1051, '636-637', 831, '952-953', '636-638', '556-557', 828, 906]
index = [165, 172, 175, '493-494', 907, 990]
index_pages = []

for num in index:
    if isinstance(num, int):
        index_pages.append(num)
    else:
        start, end = map(int, num.split('-'))
        index_pages += (list(range(start, end + 1)))


set2 = set(index_pages)

# Compare with top 10 pages
top10 = df.sort_values(by='religion', ascending=False)
top10_index = top10.index.values
print(top10_prop)

ncomponents = [100]
proportions = []
query_word = 'religion'
query_word_vec = vectorizer.transform([query_word])
for n in ncomponents:
    svd = TruncatedSVD(n_components=n, n_iter=10, random_state=42)
    Xhat = svd.fit_transform(X).dot(svd.components_)
    word_col = Xhat[:, word_idx]
    row_idx = np.argsort(word_col)[::-1]
    row_idx = row_idx[:len(index_pages)]


# Matplotlib
plt.plot(ncomponents, proportions, '-o')
# display the plot
plt.show()
