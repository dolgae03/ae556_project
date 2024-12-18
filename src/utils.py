import re, math
import xml.etree.ElementTree as ET
import Levenshtein

def parse_xml_to_segments(xml_data):
    """
    XML 데이터를 파싱하여 각 segment를 딕셔너리로 변환하고 리스트로 반환.

    Args:
        xml_data (str): XML 형식의 데이터 문자열.

    Returns:
        List[dict]: 각 segment를 표현하는 딕셔너리의 리스트.
    """
    # XML 파싱
    root = ET.fromstring(xml_data)
    segments = []

    # 데이터 추출
    for segment in root.findall('segment'):
        segment_dict = {
            'start': float(segment.find('start').text),
            'end': float(segment.find('end').text),
            'speaker': segment.find('speaker').text,
            'speaker_label': segment.find('speaker_label').text,
            'text': segment.find('text').text,
            'tags': {
                'correct': int(segment.find('tags/correct').text),
                'correct_transcript': int(segment.find('tags/correct_transcript').text),
                'correct_tagging': int(segment.find('tags/correct_tagging').text),
                'non_english': int(segment.find('tags/non_english').text),
            }
        }
        segments.append(segment_dict)
    
    return segments

def clean_text(text):
    """
    [#태그]내용[/#태그] 형태의 태그를 제거하고 내용만 남기는 함수.
    
    Args:
        text (str): 원본 문자열.
    
    Returns:
        str: 태그를 제거한 문자열.
    """
    # [#태그]와 [/#태그]를 제거하고 내용만 남김
    cleaned_text = re.sub(r'\[.*?\]', '', text)
    return cleaned_text.strip().upper()

class VocabularyMatcher:
    # WARNING: input vocabularies must be lowercased already

    def __init__(self, vocab_file):
        """
        Initialize the VocabularyMatcher class and load the vocabularies from the file.
        """
        self._load_vocab(vocab_file)
    
    def _load_vocab(self, vocab_file):
        """
        Load vocabularies from the given file into a dictionary.
        """
        vocab = {}
        vocab_max_frequency = 0
        with open(vocab_file, 'r') as f:
            for line in f:
                word, frequency = line.strip().split()
                frequency = int(frequency)
                vocab[word] = frequency
                if frequency > vocab_max_frequency: vocab_max_frequency = frequency
        
        self.vocab = vocab
        self.vocab_max_frequency = vocab_max_frequency
    
    def get_closest_word(self, input_word, weight_on_frequencies=False):
        """
        Find the word in the vocabularies closest to the given input_word using Levenshtein distance.
        """
        closest_word = None
        min_distance = float('inf')
        
        for vocab_word in self.vocab:
            distance = Levenshtein.distance(input_word.lower(), vocab_word)
            
            if weight_on_frequencies:
                vocab_word_frequency = self.vocab[vocab_word]
                vocab_word_frequency_normalized = math.log2(vocab_word_frequency / self.vocab_max_frequency)
                weight_factor = 0.4
                distance = distance - vocab_word_frequency_normalized * weight_factor

            if distance < min_distance:
                min_distance = distance
                closest_word = vocab_word
                
        return closest_word

    def get_closest_words(self, input_string, weight_on_frequencies=False):
        input_words = input_string.split(' ')
        closest_words = list(map(lambda input_word: self.get_closest_word(input_word, weight_on_frequencies), input_words))
        return ' '.join(closest_words)
