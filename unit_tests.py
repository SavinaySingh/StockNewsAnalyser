import unittest
from unittest.mock import patch, Mock, MagicMock
import pandas as pd
from bs4 import BeautifulSoup
from stock_analyser import MarketScraper, NewsAnalyzer  
from text_preprocessor import TextPreprocessor
import sys

class TestMarketScraper(unittest.TestCase):
    def setUp(self):
        # Initialize TextPreprocessor instance
        self.tp = TextPreprocessor()
        # Initialize MarketScraper instance with n=50
        scraper = MarketScraper(n=50)
        # Define URLs and market details for different stock exchanges
        urls_markets = [
            ("https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/", "INDIA", 'tablebluelink', 'tdcolumn'),
            ("https://stockanalysis.com/list/sp-500-stocks/", "USA", 'slw svelte-eurwtr', 'sym svelte-eurwtr'),
            ("https://stockanalysis.com/list/australian-securities-exchange/", "AUSTRALIA", 'slw svelte-eurwtr', 'sym svelte-eurwtr'),
            ("https://stockanalysis.com/list/london-stock-exchange/", "UNITED KINGDOM", 'slw svelte-eurwtr', 'sym svelte-eurwtr'),
            ("https://stockanalysis.com/list/toronto-stock-exchange/", "CANADA", 'slw svelte-eurwtr', 'sym svelte-eurwtr')
        ]
        # Aggregate data from the URLs and store it in a DataFrame
        df_companies = scraper.aggregate_data(urls_markets)
        self.df_companies = df_companies
        # Initialize NewsAnalyzer instance with an API key and the aggregated DataFrame
        api_key = "69cfb7a2c610444fbd7fd6705568970e"
        self.na = NewsAnalyzer(api_key, df_companies)
    
    @patch('requests.get')
    def test_fetch_market_data_india(self, mock_get):
        # Mock the response from requests.get
        mock_response = Mock()
        mock_response.text = '<html><td class="name-td"><div class="name-div"><div class="company-name">Company A</div><div class="company-code">COMPA</div></div></td></html>'
        mock_get.return_value = mock_response
        
        # Initialize MarketScraper with n=1
        scraper = MarketScraper(n=1)
        # Fetch market data using the mocked response
        result = scraper.fetch_market_data('http://example.com', 'INDIA', 'name', 'symbol')
        
        # Define the expected DataFrame
        expected = pd.DataFrame([{'Market': 'INDIA', 'Name': 'Company A', 'Code': 'COMPA.NS'}])
        # Compare the result with the expected DataFrame
        pd.testing.assert_frame_equal(result, expected)
    
    def test_parse_other_markets(self):
        # Mock HTML data
        html = '''
        <html>
            <body>
                <table>
                    <tr><td class="header">Header</td></tr>
                    <tr>
                        <td class="name">Company A</td>
                        <td class="symbol">COMPA</td>
                    </tr>
                    <tr>
                        <td class="name">Company B</td>
                        <td class="symbol">COMPB</td>
                    </tr>
                    <tr>
                        <td class="name">Company C</td>
                        <td class="symbol">COMPC</td>
                    </tr>
                </table>
            </body>
        </html>
        '''
        
        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        # Initialize MarketScraper with n=2
        scraper = MarketScraper(n=2)
        # Parse the market data from the mocked HTML
        result = scraper._parse_other_markets(soup, 'Other', 'name', 'symbol')
        
        # Define the expected DataFrame
        expected = pd.DataFrame({
            "Market": ["OTHER", "OTHER"], 
            'Name': ["Company A", "Company B"], 
            'Code': ["COMPA", "COMPB"]
        })
        
        # Compare the result with the expected DataFrame
        pd.testing.assert_frame_equal(result, expected)
    
    def test_remove_punctuation(self):
        text = "Hello, world! This is a test."
        # Test the remove_punctuation method
        result = self.tp.remove_punctuation(text)
        self.assertEqual(result, "Hello  world  This is a test ")
    
    def test_decontracted(self):
        phrase = "He won't go to the park."
        # Test the decontracted method
        result = self.tp.decontracted(phrase)
        self.assertEqual(result, "He will not go to the park.")
    
    def test_lemmatize_text(self):
        text = "running eating cats"
        # Test the lemmatize_text method
        result = self.tp.lemmatize_text(text)
        self.assertEqual(result, "run eat cat")
    
    def test_preprocessing(self):
        sentences = ["This is a test sentence.", "Another test sentence with numbers 123."]
        # Test the preprocessing method
        result = self.tp.preprocessing(sentences)
        self.assertEqual(result, ["test sentence", "another test sentence number"])
    
    def test__analyze_sentiment_positive(self):
        text = "This is a positive sentence."
        # Test the _analyze_sentiment method for positive sentiment
        result = self.na._analyze_sentiment(text)
        self.assertEqual(result, 'Positive')
    
    def test__analyze_sentiment_negative(self):
        text = "This is a negative sentence."
        # Test the _analyze_sentiment method for negative sentiment
        result = self.na._analyze_sentiment(text)
        self.assertEqual(result, 'Negative')
    
    def test__analyze_sentiment_neutral(self):
        text = "This is a neutral sentence."
        # Test the _analyze_sentiment method for neutral sentiment
        result = self.na._analyze_sentiment(text)
        self.assertEqual(result, 'Neutral')
    
    @patch('sys.argv', ['stock_analyser.py', 'SAMPLE_API_KEY'])
    def test_init_with_api_key(self):
        with patch('stock_analyser.TextPreprocessor') as mock_text_preprocessor:
            api_key = sys.argv[1]
            # Initialize NewsAnalyzer with the API key from sys.argv
            news_analyzer = NewsAnalyzer(api_key, self.df_companies)
            # Check that the TextPreprocessor was initialized
            mock_text_preprocessor.assert_called_once()
            # Check that the API key was set correctly
            self.assertEqual(news_analyzer._api_key, api_key)
    
    @patch('stock_analyser.YouTubeTranscriptApi.get_transcript')
    @patch('requests.get')
    def test__extract_content(self, mock_requests_get, mock_get_transcript):
        # Mock the response from requests.get for an article
        mock_requests_get.return_value = MagicMock(content='<html><body><article>This is an article</article></body></html>')
        # Sample data dictionary similar to what the method receives
        sample_data = {'articles': [{'url': 'https://www.example.com/article1'}, {'url': 'https://www.youtube.com/article2'}]}
        # Call the _extract_content method
        result_content, result_content_type_dict = self.na._extract_content(sample_data)
        # Assert that the result is as expected
        self.assertEqual(len(result_content), 2)
        self.assertIn('https://www.example.com/article1', result_content)
        self.assertIn('https://www.youtube.com/article2', result_content)
        self.assertEqual(result_content['https://www.example.com/article1'], 'This is an article')
        self.assertEqual(len(result_content_type_dict), 2)
        self.assertIn('https://www.example.com/article1', result_content_type_dict)
        self.assertIn('https://www.youtube.com/article2', result_content_type_dict)
        self.assertEqual(result_content_type_dict['https://www.example.com/article1'], 'ARTICLE')
      
   
    
if __name__ == '__main__':
    unittest.main()
