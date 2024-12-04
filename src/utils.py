import xml.etree.ElementTree as ET
import re, torch


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
    return cleaned_text.strip()