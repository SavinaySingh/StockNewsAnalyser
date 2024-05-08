import unittest
from unittest.mock import patch
from news_analyser import NewsAnalyzer
import sys
from unittest.mock import patch, MagicMock
from text_preprocessor import TextPreprocessor

class TestNewsAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.tp = TextPreprocessor()
        self.na=NewsAnalyzer("69cfb7a2c610444fbd7fd6705568970e")

    def test_remove_punctuation(self):
        text = "Hello, world! This is a test."
        result = self.tp.remove_punctuation(text)
        self.assertEqual(result, "Hello  world  This is a test ")

    def test_decontracted(self):
        phrase = "He won't go to the park."
        result = self.tp.decontracted(phrase)
        self.assertEqual(result, "He will not go to the park.")

    def test_lemmatize_text(self):
        text = "running eating cats"
        result = self.tp.lemmatize_text(text)
        self.assertEqual(result, "run eat cat")

    def test_preprocessing(self):
        sentences = ["This is a test sentence.", "Another test sentence with numbers 123."]
        result = self.tp.preprocessing(sentences)
        self.assertEqual(result, ["test sentence", "another test sentence number"])

    
    @patch('sys.argv', ['news_analyser.py', 'SAMPLE_API_KEY'])
    def test_init_with_api_key(self):
        with patch('news_analyser.TextPreprocessor') as mock_text_preprocessor:
            api_key = sys.argv[1]
            news_analyzer = NewsAnalyzer(api_key)
            # Check that the TextPreprocessor was initialized
            mock_text_preprocessor.assert_called_once()

            # Check that the API key was set correctly
            self.assertEqual(news_analyzer._api_key, api_key)
            
    
    @patch('news_analyser.YouTubeTranscriptApi.get_transcript')
    @patch('requests.get')
    def test__extract_content(self,mock_requests_get,mock_get_transcript):
        
        mock_requests_get.return_value = MagicMock(content='<html><body><article>This is an article</article></body></html>')
        # Create a sample data dictionary similar to what your method receives
        sample_data = {'articles': [{'url': 'https://www.example.com/article1'}, {'url': 'https://www.youtube.com/article2'}]}
        # Call the method you want to test
        result_content, result_content_type_dict = self.na._extract_content(sample_data)
        
        # Call the method you want to test
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
        
        
    def test__analyze_sentiment_positive(self):
        text = "This is a positive sentence."
        result = self.na._analyze_sentiment(text)
        self.assertEqual(result, 'Positive')

    def test__analyze_sentiment_negative(self):
        text = "This is a negative sentence."
        result = self.na._analyze_sentiment(text)
        self.assertEqual(result, 'Negative')

    def test__analyze_sentiment_neutral(self):
        text = "This is a neutral sentence."
        result = self.na._analyze_sentiment(text)
        self.assertEqual(result, 'Neutral')
      
   
    
if __name__ == '__main__':
    unittest.main()