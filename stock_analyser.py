import sys
import tkinter as tk
from tkinter import ttk, scrolledtext
from wordcloud import WordCloud
from textblob import TextBlob
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import string
import re
import requests
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from text_preprocessor import TextPreprocessor
from youtube_transcript_api import YouTubeTranscriptApi
import yfinance as yf

class MarketScraper:
    def __init__(self, n=50):
        self.n = n
        self.df_companies = pd.DataFrame()

    def fetch_market_data(self, url, market, name, symbol):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        if market.upper() == "INDIA":
            return self._parse_indian_market(soup, market)
        elif market.upper() in ("USA", "AUSTRALIA", "UNITED KINGDOM", "CANADA"):
            return self._parse_other_markets(soup, market, name, symbol)
        else:
            raise ValueError("Invalid market value. Please provide one of: 'INDIA', 'USA', 'AUSTRALIA', 'UNITED KINGDOM', 'CANADA'.")

    def _parse_indian_market(self, soup, market):
        td_elements = soup.find_all('td', class_='name-td')
        company_data = []
        for td in td_elements[:self.n]:
            name_div = td.find('div', class_='name-div')
            company_name = name_div.find('div', class_='company-name').text.strip()
            company_code = name_div.find('div', class_='company-code').text.strip()
            
            if company_code.endswith(".BSE"):
                company_code = company_code.replace(".NS", ".BSE")
            elif not company_code.endswith(".NS"):
                company_code += ".NS"
                
            company_data.append({"Market": market.upper(), 'Name': company_name, 'Code': company_code})

        return pd.DataFrame(company_data)

    def _parse_other_markets(self, soup, market, name, symbol):
        rows = soup.find_all('tr')
        symbols = []
        names = []

        for row in rows[1:self.n+1]:
            namedata = row.find('td', {'class': name})
            symboldata = row.find('td', {'class': symbol})
            
            if symboldata and namedata:
                symbols.append(symboldata.text.strip())
                names.append(namedata.text.strip())

        return pd.DataFrame({"Market": market.upper(), 'Name': names, 'Code': symbols})

    def aggregate_data(self, urls_markets):
        for url, market, name, symbol in urls_markets:
            df_market = self.fetch_market_data(url, market, name, symbol)
            self.df_companies = pd.concat([self.df_companies, df_market], ignore_index=True)
        return self.df_companies

class NewsAnalyzer:
    def __init__(self,api_key,df_companies):
        self.text_preprocessor=TextPreprocessor()
        df = pd.DataFrame()
        self._api_cache = {}
        self._web_cache = {}
        self._api_key = api_key
        self.df_companies = df_companies

    def get_companies_list(self, event=None):
        selected_country = self.country_combobox.get()
        companies = self.df_companies.loc[self.df_companies["Market"] == selected_country, "Name"].values.tolist()        
        self.company_combobox.config(values=companies)
        self.company_combobox.current(0)    
        
    def fetch_articles(self):
        selected_company = self.company_combobox.get()
        
        base_url = 'https://newsapi.org/v2/everything?'
        params = {
            'q': selected_company,
            'apiKey': self._api_key,
            'pageSize': int(self.article_count_var.get())
        }
        cache_key = tuple(sorted(params.items()))  
        if cache_key in self._api_cache:
            data = self._api_cache[cache_key]
        else:
            response = requests.get(base_url, params=params)
            data = response.json()
            self._api_cache[cache_key] = data
        
        response = requests.get(base_url, params=params)
        data = response.json()

        content, content_type_dict = self._extract_content(data)
        self.article_listbox.delete(0, tk.END)
        articles = []
        for article in data['articles']:
            articles.append({
                'source': article['source'],
                'author': article['author'],
                'title': article['title'],
                'description': article['description'],
                'url': article['url'],
                'urlToImage': article['urlToImage'],
                'publishedAt': article['publishedAt'],
                'content': article['content']
            })
        
        NewsAnalyzer.df = pd.DataFrame(articles)
        NewsAnalyzer.df['Content_Type'] = NewsAnalyzer.df['url'].map(content_type_dict)

        NewsAnalyzer.df['Content_Text'] = NewsAnalyzer.df['url'].map(content)
        NewsAnalyzer.df.loc[NewsAnalyzer.df.Content_Type.isna(), 'Content_Type'] = ''
        NewsAnalyzer.df.loc[NewsAnalyzer.df.Content_Text.isna(), 'Content_Text'] = ''
        NewsAnalyzer.df['Content_Text_Cleaned'] = self.text_preprocessor.preprocessing(list(NewsAnalyzer.df.Content_Text))
        NewsAnalyzer.df['Sentiment'] = [self._analyze_sentiment(x) for x in NewsAnalyzer.df['Content_Text']]

        for i in range(0, len(NewsAnalyzer.df)):
            self.article_listbox.insert(tk.END, f"{NewsAnalyzer.df['title'][i]} -- CONTENT_TYPE: {NewsAnalyzer.df['Content_Type'][i]} -- Sentiment: {NewsAnalyzer.df['Sentiment'][i]} ")
            if NewsAnalyzer.df['Sentiment'][i] == 'Positive':
                self.article_listbox.itemconfig(i, {'fg': 'green'})
            elif NewsAnalyzer.df['Sentiment'][i] == 'Negative':
                self.article_listbox.itemconfig(i, {'fg': 'red'})

    def _extract_content(self, data):
        content = {}
        content_type_dict = {}

        def extract_video_id(url):
            pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
            match = re.search(pattern, url)
            if match:
                return match.group(1)
            return None

        for article in tqdm(data['articles'], total=len(data['articles'])):
            url = article['url']
            
            if url in self._web_cache:
                article_content, content_type = self._web_cache[url]
            else:
                response = requests.get(url, allow_redirects=True)
                if 'youtube.com' in response.url:
                    content_type = 'YOUTUBE VIDEO'
                    video_url = response.url
                    video_id = extract_video_id(video_url)
                    try:
                        transcript = YouTubeTranscriptApi.get_transcript(video_id)
                        article_content = ' '.join([x['text'] for x in transcript])
                    except Exception as e:
                        error_message = str(e)
                        if 'Transcripts are disabled' in error_message:
                            article_content = "Transcripts are disabled for this video."
                        elif 'Video unavailable' in error_message:
                            article_content = "The video is unavailable."
                        elif 'No transcript' in error_message:
                            article_content = "No transcript is available for this video."
                        else:
                            article_content = f"An error occurred: {error_message}"
                else:
                    content_type = 'ARTICLE'
                    soup = BeautifulSoup(response.content, 'html.parser')
                    article_content = ''
                    article_tag = soup.find('article')
                    if article_tag:
                        article_content = article_tag.text.strip()
                    else:
                        article_content = ' '.join([x.text.strip() for x in soup.find_all('p')])
                self._web_cache[url] = (article_content, content_type)

            content[url] = article_content
            content_type_dict[url] = content_type

        return content, content_type_dict

    def _analyze_sentiment(self, text):
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            return 'Positive'
        elif sentiment < 0:
            return 'Negative'
        else:
            return 'Neutral'

    def _plot_sentiment(self):
        sentiment_counts = NewsAnalyzer.df['Sentiment'].value_counts()
        source_counts = pd.Series([x['name'] for x in NewsAnalyzer.df['source']]).value_counts()

        sentiment_window = tk.Toplevel()
        sentiment_window.title('Dashboard')
        sentiment_window.geometry('2000x1000')

        fig = plt.figure(figsize=(14, 8))

        ax1 = fig.add_subplot(223)
        ax1.pie(sentiment_counts, labels=sentiment_counts.index, startangle=90, autopct='%1.1f%%')
        ax1.axis('equal')

        ax2 = fig.add_subplot(224)
        ax2.axis("off")
        ax2.set_title('Word Cloud for Articles')
        all_text_cleaned = ' '.join(NewsAnalyzer.df['Content_Text_Cleaned'].dropna())
        if all_text_cleaned:
            wordcloud = WordCloud(width=1200, height=800, background_color='white').generate(all_text_cleaned)
            ax2.imshow(wordcloud, interpolation='bilinear')

        vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words='english')
        X = vectorizer.fit_transform(NewsAnalyzer.df['title'])

        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(X)

        feature_names = vectorizer.get_feature_names_out()
        topic_names = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-5:-1]]
            topic_name = ','.join(top_words)
            topic_names.append(topic_name)

        NewsAnalyzer.df['topic'] = lda.transform(X).argmax(axis=1)
        NewsAnalyzer.df['topic_name'] = NewsAnalyzer.df['topic'].map(lambda x: topic_names[x])

        sid = SentimentIntensityAnalyzer()
        NewsAnalyzer.df['sentiment_score'] = NewsAnalyzer.df['title'].apply(lambda x: sid.polarity_scores(x)['compound'])
        topic_sentiments = NewsAnalyzer.df.groupby('topic_name')['sentiment_score'].mean()

        ax3 = fig.add_subplot(221)
        ax3.bar(range(len(topic_sentiments)), topic_sentiments, color='skyblue')
        ax3.set_xlabel('Topic')
        ax3.set_ylabel('Average Sentiment Score')
        ax3.set_title('Sentiment Distribution by Topic')
        ax3.set_xticks(range(len(topic_sentiments)))
        ax3.set_xticklabels(topic_sentiments.index, rotation=45)

        ax4 = fig.add_subplot(222)
        sns.barplot(x=source_counts.index, y=source_counts.values, ax=ax4)
        ax4.set_xticklabels(source_counts.index, rotation=90, ha='right')
        ax4.set_xlabel('Source')
        ax4.set_ylabel('Number of Articles')
        ax4.set_title('Number of Articles by Source')

        plt.subplots_adjust(hspace=1, wspace=0.3)

        canvas = FigureCanvasTkAgg(fig, master=sentiment_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def _show_content(self, event):
        index = self.article_listbox.curselection()[0]
        self.content_text.delete('1.0', tk.END)
        self.content_text.insert(tk.END, NewsAnalyzer.df['Content_Text'][index])


# Function to get the list of companies based on the country selected
    def get_companies_list(self, event=None):
        selected_country = self.country_combobox.get()
        companies = self.df_companies.loc[self.df_companies["Market"] == selected_country, "Name"].values.tolist()        
        self.company_combobox.config(values=companies)
        self.company_combobox.current(0)

# Get the company and country related data
    def get_company(self):
        self.selected_company = self.company_combobox.get()
        selected_country = self.country_combobox.get()
        company = self.df_companies.loc[self.df_companies["Name"] == self.selected_company, "Code"].values.tolist()

        # Create country-specific variables like exchange postfixes and currency
        if selected_country == "INDIA":
            company_code = str(company[0])
            currency = "INR"
        elif selected_country == "AUSTRALIA":
            company_code = str(company[0]) + ".AX"
            currency = "AUD"
        elif selected_country == "UNITED KINGDOM":
            company_code = company[0] + ".L"
            currency = "GBP"
        elif selected_country == "CANADA":
            company_code = company[0] + ".TO"
            currency = "CAD"
        else:
            company_code = str(company[0])
            currency = "USD"
            
        df = yf.download(tickers=company_code, period='1y', interval='1d')
        self.print_share_price(df, company_code, self.selected_company, currency)

# Graph the histrocial share price of the stock
    def print_share_price(self, df, stock, name, currency):
        # Clear previous graph
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        # Plot candlestick chart using Matplotlib
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(df.index, df['Open'], color='black', label='Open')
        ax.plot(df.index, df['High'], color='green', label='High')
        ax.plot(df.index, df['Low'], color='red', label='Low')
        ax.plot(df.index, df['Close'], color='blue', label='Close')
        ax.set_title(f"{name} Share Price")
        ax.set_ylabel(f"Stock Price ({currency} per Share)")
        ax.legend()
        ax.grid(True)

        # Embed the Matplotlib figure into Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        
    def run(self):
        root = tk.Tk()
        root.title("Stock Analyser")
        root.geometry("2000x1000")


        canvas = tk.Canvas(root)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=canvas_frame, anchor=tk.NW)

        canvas_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"), yscrollcommand=scrollbar.set))

        main_frame = ttk.Frame(canvas_frame, padding="60")
        main_frame.pack()

        # List of countries
        self.Countries = list(set(df_companies["Market"].tolist()))

        # Create label for country selection        
        country_label = ttk.Label(main_frame, text="Select a country:")
        country_label.grid(row=0, column=0, sticky="w")
        self.country_combobox = ttk.Combobox(main_frame, values=self.Countries, state="readonly")
        self.country_combobox.grid(row=0, column=1, padx=10)
        self.country_combobox.bind("<<ComboboxSelected>>", self.get_companies_list)
        self.country_combobox.current(0)


        # Label for company selection

        company_label = ttk.Label(main_frame, text="Select a company:")
        company_label.grid(row=1, column=0, sticky="w")
        self.company_combobox = ttk.Combobox(main_frame, state="readonly")
        self.company_combobox.grid(row=1, column=1, padx=10)
        self.company_combobox.set("")  # Default value

        article_count_label = ttk.Label(main_frame, text="Number of articles:")
        article_count_label.grid(row=2, column=0, sticky="w")
        self.article_count_var = tk.StringVar()
        self.article_count_var.set("10")
        article_count_entry = ttk.Spinbox(main_frame, from_=1, to=50, textvariable=self.article_count_var, width=5)
        article_count_entry.grid(row=2, column=1, padx=10)

        fetch_button = ttk.Button(main_frame, text="Fetch Articles", command=self.fetch_articles)
        fetch_button.grid(row=0, column=2, rowspan=2, padx=10)

        get_share_price_button = ttk.Button(main_frame, text="Get Share Price", command=self.get_company)
        get_share_price_button.grid(row=1, column=2, rowspan=2, padx=10)

        # Split main_frame into two parts
        self.article_frame = ttk.Frame(main_frame)
        self.article_frame.grid(row=3, column=0, columnspan=3, padx = 2, pady=10,sticky="nsew")

        self.graph_frame = ttk.Frame(main_frame)
        self.graph_frame.grid(row=3, column=2, rowspan=2, padx=8, pady=10,sticky="nsew")

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.columnconfigure(3, weight=1)
        main_frame.columnconfigure(4, weight=1)
        main_frame.columnconfigure(5, weight=1)
        
        self.article_listbox = tk.Listbox(main_frame, width=70, height=20)
        self.article_listbox.grid(row=3, column=0, columnspan=1, pady=10,sticky="nsew")

        scrollbar_listbox = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.article_listbox.yview)
        scrollbar_listbox.grid(row=3, column=3, sticky="ns")
        self.article_listbox.config(yscrollcommand=scrollbar_listbox.set)

        self.article_listbox.bind('<<ListboxSelect>>', self._show_content)

        self.content_text = scrolledtext.ScrolledText(main_frame, wrap='word', height=15, width=180)
        self.content_text.grid(row=6, columnspan=3, pady=10)

        plot_button = ttk.Button(main_frame, text="More Information", command=self._plot_sentiment)
        plot_button.grid(row=7, column=0, columnspan=3, pady=10)

        root.mainloop()


if __name__ == "__main__":
    urls_markets = [
        ("https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/", "INDIA", 'tablebluelink', 'tdcolumn'),
        ("https://stockanalysis.com/list/sp-500-stocks/", "USA", 'slw svelte-eurwtr', 'sym svelte-eurwtr'),
        ("https://stockanalysis.com/list/australian-securities-exchange/", "AUSTRALIA", 'slw svelte-eurwtr', 'sym svelte-eurwtr'),
        ("https://stockanalysis.com/list/london-stock-exchange/", "UNITED KINGDOM", 'slw svelte-eurwtr', 'sym svelte-eurwtr'),
        ("https://stockanalysis.com/list/toronto-stock-exchange/", "CANADA", 'slw svelte-eurwtr', 'sym svelte-eurwtr')
    ]

    scraper = MarketScraper(n=50)
    df_companies = scraper.aggregate_data(urls_markets)
    if len(sys.argv) < 2:
        print("Usage: python3 x.py <apikey>")
        sys.exit(1)
    api_key = sys.argv[1]
    news_analyzer = NewsAnalyzer(api_key,df_companies)
    news_analyzer.run()


