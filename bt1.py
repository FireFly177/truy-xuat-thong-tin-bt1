from selenium import webdriver
from selenium.webdriver.chrome.service import Service  # Import the Service class
from bs4 import BeautifulSoup
import time
import re
import nltk
from collections import Counter
from nltk.util import ngrams
import spacy

# Ensure required NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Initialize spaCy for NER
nlp = spacy.load("en_core_web_sm")

# Function to crawl and clean the text with improved cleaning rules using Selenium
def crawl_and_clean(url):
    # Set up the WebDriver (make sure to provide the correct path to your chromedriver)
    driver_path = './chromedriver-win64/chromedriver.exe'  # Update this path
    service = Service(executable_path=driver_path)  # Use Service for specifying the path
    driver = webdriver.Chrome(service=service)
    
    # Open the URL
    driver.get(url)
    
    # Wait for the page to load (you can adjust the time depending on the website)
    time.sleep(5)  # Wait for 5 seconds for content to load
    
    # Get the page source (HTML)
    html_content = driver.page_source
    
    # Close the driver
    driver.quit()
    
    # Now we can parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()

    # Remove unwanted characters and noise
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s,.!?]', '', text)  # Keep letters, numbers, and selected punctuation
    text = text.strip()
    
    return text

# Function to count words, sentences, and paragraphs
def count_text_elements(text):
    words = len(text.split())
    sentences = len(re.findall(r'[.!?]', text))  # Basic sentence split
    paragraphs = len(text.split('\n\n'))  # Paragraphs are assumed to be separated by blank lines
    return words, sentences, paragraphs

# Function to compute vocabulary and word frequency
def compute_vocabulary(text):
    words = text.split()
    word_freq = Counter(words)
    return word_freq

# Function to remove stopwords and create a new vocabulary
stop_words = set(nltk.corpus.stopwords.words('english'))  # Customize for Vietnamese stopwords if needed

def remove_stopwords_and_generate_vocab(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    new_vocab = Counter(filtered_words)
    return new_vocab

# Function to extract unigrams, bigrams, and trigrams
def extract_ngrams(text):
    words = text.split()
    unigrams = list(ngrams(words, 1))
    bigrams = list(ngrams(words, 2))
    trigrams = list(ngrams(words, 3))
    return unigrams, bigrams, trigrams

# Function to extract all anchor text
def extract_anchor_text(url):
    driver_path = './chromedriver-win64/chromedriver.exe'  # Update this path
    service = Service(executable_path=driver_path)  # Use Service for specifying the path
    driver = webdriver.Chrome(service=service)
    driver.get(url)
    time.sleep(5)  # Wait for 5 seconds to allow the page to load
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    anchors = soup.find_all('a')
    anchor_texts = [anchor.get_text() for anchor in anchors if anchor.get_text()]
    driver.quit()
    return anchor_texts

# Function for Named Entity Recognition (NER) using spaCy or HuggingFace pipeline
def perform_ner(text):
    # Using spaCy for NER
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Crawl two random Vietnamese websites
url1 = "https://dantri.com.vn/giao-duc/tphcm-mien-hoc-phi-cho-toan-bo-hoc-sinh-tu-nam-hoc-toi-20250220174750763.htm"
url2 = "https://vnexpress.net/gia-tien-ao-pi-nhay-mua-sau-khi-mo-mang-4852011.html"

# Crawl and clean text from the websites
text1 = crawl_and_clean(url1)
text2 = crawl_and_clean(url2)

# Task 1: Clean text
print("Cleaned Text 1: ", text1)  # Preview of the first 500 characters
print("Cleaned Text 2: ", text2)  # Preview of the first 500 characters

# Task 2: Count words, sentences, and paragraphs
words1, sentences1, paragraphs1 = count_text_elements(text1)
words2, sentences2, paragraphs2 = count_text_elements(text2)
print(f"Text 1 - Words: {words1}, Sentences: {sentences1}, Paragraphs: {paragraphs1}")
print(f"Text 2 - Words: {words2}, Sentences: {sentences2}, Paragraphs: {paragraphs2}")

# Task 3: Create initial vocabulary and compute word frequency
vocab1 = compute_vocabulary(text1)
vocab2 = compute_vocabulary(text2)
print(f"Vocabulary 1 (sample): {list(vocab1.items())[:10]}")
print(f"Vocabulary 2 (sample): {list(vocab2.items())[:10]}")

# Task 4: Remove stopwords and generate a new vocabulary (V2)
new_vocab1 = remove_stopwords_and_generate_vocab(text1)
new_vocab2 = remove_stopwords_and_generate_vocab(text2)
print(f"New Vocabulary 1 (sample): {list(new_vocab1.items())[:10]}")
print(f"New Vocabulary 2 (sample): {list(new_vocab2.items())[:10]}")

# Task 5: Extract unigrams, bigrams, and trigrams
unigrams1, bigrams1, trigrams1 = extract_ngrams(text1)
unigrams2, bigrams2, trigrams2 = extract_ngrams(text2)
print(f"Unigrams from Text 1: {unigrams1[:5]}")
print(f"Bigrams from Text 1: {bigrams1[:5]}")
print(f"Trigrams from Text 1: {trigrams1[:5]}")

# Task 6: Extract anchor text from the crawled documents
anchors1 = extract_anchor_text(url1)
anchors2 = extract_anchor_text(url2)
print(f"Anchor text from Text 1: {anchors1[:5]}")
print(f"Anchor text from Text 2: {anchors2[:5]}")

# Task 7: Perform Named Entity Recognition (NER)
entities1 = perform_ner(text1)
entities2 = perform_ner(text2)
print(f"Entities from Text 1: {entities1[:5]}")
print(f"Entities from Text 2: {entities2[:5]}")
