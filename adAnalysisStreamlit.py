import streamlit as st
# Data Structure
import pandas as pd
import numpy as np
from ast import literal_eval
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Text Analysis
from wordcloud import WordCloud
from collections import Counter, defaultdict
from konlpy.tag import Okt
import re

# Vision
from PIL import Image

# System
import io
import os
from io import BytesIO

# Access the internet
from urllib.request import urlopen
import requests
import os

#os.chdir('C:\\Users\\7info\\Desktop\\Content_Evaluation')
# 데이터 로딩

def plot_wordcloud(df, text_feature, font_path='BMDOHYEON_ttf.ttf'):
    plt.rc('font', family='Malgun Gothic')
    new_df = df.dropna(subset=[text_feature])
    text = ' '.join(new_df[text_feature])
    wordcloud = WordCloud(width=1200, height=800, background_color='white', font_path=font_path).generate(text)
    plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    
def calculate_token_scores(df, text_feature):
    # 각 토큰 별로 score 값을 저장할 딕셔너리 생성
    new_df = df.dropna(subset=[text_feature])
    token_scores = defaultdict(list)

    # df의 각 행을 순회하면서
    for idx, row in new_df.iterrows():
        tokens = row[text_feature]  # 해당 행의 토큰 리스트를 가져옴
        score = row['score']  # 해당 행의 score 값을 가져옴
        # 토큰이 None이면 무시
        try:
            for token in tokens:  
                token_scores[token].append(score)  # 해당 토큰의 score 리스트에 현재 score 추가
        except:
            continue

    # 각 토큰 별로 score의 평균을 계산
    token_avg_scores = {token: round(sum(scores) / len(scores),1) for token, scores in token_scores.items()}
    df_token_scores = pd.DataFrame(list(token_avg_scores.items()), columns=['Token', 'Average Score'])
    df_token_scores.sort_values('Average Score', ascending=False, inplace=True)
    df_token_scores.reset_index(drop=True, inplace=True)
    return df_token_scores



def remove_digit_and_single_char(text):
    cleaned_text = re.sub(r'\d+[가-힣]', '', text)
    return cleaned_text

# 형태소 분석하여, 제목을 토큰화 하는 함수
def tokenize(text):
    if pd.isna(text):
        return None
    okt = Okt()
    stopwords = ['/', '[', ']', '+', '-', '_', '=', '(', ')', '{', '}',
                 '>', '<', ':', ';', '.', ',', '?', '!', '@', '#', '$',
                 '%', '^', '&', '*', '...','"', "''"]
    
    # 제거해야 하는 지시대명사
    pronouns = ["이","여기","그", "이", "저", "아무", "무엇", "어디", "언제", "누구", "그거", "이렇게", "때", "얘", "니", "제","네가","거", "이거", "내", "이번", "너", "나", "어느", "것"]
    # 제거해야 하는 의미없는 단어
    meaningless = ["좀","데" , "뭐", "듯", "머", "뿐", "하다", "수", "어쩌", "온", "안", "다", "편" "너무", "요", "것", "더", "왜", "는가", "걸", "함",
                   "은", "이다", "있다", "게", "후", "막", "근데", "딱", "쪽", "과연", "속", "및", "뭘", "당신", "의", "마", "날", "량", "어쩌나", "또",
                   "이젠", "놈", "세"]
    # 일반적으로 제거하듯이 고유명사를 제거해야 하나?
#     proper_nouns =["율지문덕", "엔틱보스","문현빈", "민이네다육", "05학번이즈히어", "웨스트브룩", "서조", "가오나시티", "웰리힐리파크", "터틀비치", "엔틱보스", "만만사단", "피린이", "코-액시얼 마스터 크로노미터 크로노그래프", "수부지", "우드샷", "삼성전자", "앙스타", "롤토체스", "실비집", "가시와다육이마을", "공미", "포토라이", "헤라나스", "태그라크", "세븐나이츠레볼루션", "다육풍경", "젤버스트", "칼리스타", "에리카", "한문철", "찐시황","레이커스", "에스파", "코나", "셀토스", "수삼TV", "쿠네리뷰", "박군", "리뷰대장", "필아이비", "클체", "풀문양", "리리코", "전섭", "세인트릴리", "왕만이형", "봉희악", "부캐릭", "에픽세븐", "삼양", "키나", "꼰보석", "로벅스", "신라젠", "적혈공성선", "용느사조직", "뮨법사","업비트","김톤슨"]    
    # 쪼갰을 때 의미가 손상되는 단어
#     corruption_words = ["가성비","방향지시등", "구독자", "웃음벨", "꼰대", "라이징", "토르템","알트코인", "브이로그", "미니멀","일반통행","성장주", "영입", "헬로우", "퀸아망", "신화인형", "재상장", "스펙업", "토르템", "각인템", "수입차", "드림카", "올드카", "축캐릭", "양측", "대항마","찐", "리먼사태", "참가비", "초강력", "포메이션", "리프레쉬", "통모짜", "시음회", "아우터", "데일리룩", "출근룩", "상자깡", "올타임", "갓템", "동호인", "살림템", "데일리", "자율주행", "낄스윗홈", "출조점", "방구석토크","초간단", "비하인드", "백패킹", "클로징벨", "한방", "저점", "자작극", "전세캠", "페이스리프트", "에어프라이어","꾸안꾸", "무과금","리뷰", "꿀조합","콤백홈", "투핸드", "킹피스", "템트리","혼조세", "베어마켓랠리"]

#     # 본격적으로 전처리를 하기전 쪼개지 말아야 하는 단어에 해당하는 단어를 저장하는 리스트
#     keep_words = []
#     # 단어를 쪼개기 전 corruption_words에 해당하는 단어들이 문장 내 있으면 추출
#     for corruption_word in corruption_words:
#         if corruption_word in text:
#             keep_words.append(corruption_word)
#             text = text.replace(corruption_word, " ")
    # 2307회등 (숫자+명사)된 경우 전체 제거         
    text=remove_digit_and_single_char(text)
    # 정규화 및 어근 추출
    words = okt.pos(text, norm=True, stem=True)  
    # 품사 태그 선택
    words = [word for word, pos in words if pos in ['Noun', 'Adjective', 'Verb', 'Adverb']]  # 선택한 품사만 추출
    # 불용어, 지시대명사, 의미없는 단어, 고유명사 제거
    words = [word for word in words if word not in stopwords + pronouns + meaningless] 
    # 최종 단어와 사전에 분리해둔 쪼개지 말아야 하는 단어 리스트를 합한다.
    # words = words + keep_words
    
    return words

def plot_freq_keyword(df, text_feature):
    plt.rc('font', family='Malgun Gothic')
    new_df = df.dropna(subset=[text_feature])
    df_token_scores = calculate_token_scores(new_df, text_feature)
    df_token_scores.set_index('Token', inplace=True)
    
    words = [item for sublist in new_df[text_feature].tolist() for item in sublist]
    # 단어의 빈도수 계산
    word_freq = Counter(words)
    # 상위 10개 단어만 선택
    top_10 = word_freq.most_common(10)
    # 데이터프레임 생성
    word_df = pd.DataFrame(top_10, columns=['Word', 'Frequency'])
    # 막대 그래프 그리기
    plt.figure(figsize=(12, 5))
    word_df.plot(kind='bar', x='Word', y='Frequency', color='orange')
    plt.xticks(rotation=0)
    plt.xlabel("")
    plt.yticks([])  # Hide the x-axis tick labels
    plt.legend().set_visible(False)
    st.pyplot(plt)
    result_df = df_token_scores.loc[word_df['Word']].T
    result_df.index = ['평균 점수']
    return result_df
    # top_10에 해당하는 토큰들의 평균 점수 출력

def convert_time(minutes):
    minutes_int = int(minutes)
    seconds = int((minutes - minutes_int) * 60)
    time_str = f"{minutes_int}분 {seconds}초"
    return time_str

def make_one_str(df, feature):
    tmp_df = df.dropna(subset=feature)
    all_tags = []
    list_col = tmp_df[feature].apply(literal_eval)
    for tag_list in list_col:
        all_tags.extend(tag_list)
        
    all_tags_str = ' '.join(all_tags)
    all_tags_str = all_tags_str.replace('&', ' ')
    return all_tags, all_tags_str

def color_print(color_df, genre_eng, feature):
    color_value = color_df[color_df['genre'] == genre_eng][feature]
    color_value = color_value.iloc[0]
    if color_value == "Very High":
        return "매우 높아요"
    elif color_value == "High":
        return "높아요"
    elif color_value == "Medium":
        return "낮아요"
    elif color_value == "Low":
        return "매우 낮아요"  
    
# df = pd.read_csv('good_ad_data.csv')
# color_df = pd.read_csv('good_ad_color.csv')
df = pd.read_csv('https://raw.githubusercontent.com/Hasaero/Content-Evaluation-Model/master/good_ad_data.csv')
color_df = pd.read_csv('https://raw.githubusercontent.com/Hasaero/Content-Evaluation-Model/master/good_ad_color.csv')
df['title_token'] = df['title'].apply(tokenize)
df['thumbnail_text_token'] = df['thumbnail_text'].apply(tokenize)
# 한글화를 위한 장르 딕셔너리
genre_dict = {
#'동물': 'Pets & Animals',
 '자동차': 'Autos & Vehicles',
 '일상': 'People & Blogs',
 '방법 & 스타일': 'Howto & Style',
 '여행': 'Travel & Events',
 '음악': 'Music',
 '게임': 'Gaming',
 '교육': 'Education',
 #'과학 & 기술': 'Science & Technology',
 '엔터테인먼트': 'Entertainment',
 '코미디': 'Comedy',
 '스포츠': 'Sports',
 #'뉴스 & 정치': 'News & Politics',
 #'영화 & 애니메이션': 'Film & Animation'
 }

emoticon_dict = {'동물': '🐶',
 '자동차': '🚓',
 '일상': '😀',
 '방법 & 스타일': '🪞',
 '여행': '🛫',
 '음악': '🎵',
 '게임': '🎮',
 '교육': '🏫',
 '과학 & 기술': '👨‍💻',
 '엔터테인먼트': '📺',
 '코미디': '😂',
 '스포츠': '⚽',
 '뉴스 & 정치': '📰',
 '영화 & 애니메이션': '🍿'}

# 장르 선택 (한글로 표시)
if 'genre' not in st.session_state:
    st.session_state['genre'] = None

page = st.sidebar.selectbox("어떤 특징을 찾으시나요?", ['홈', '어떤 제목이 인기가 많을까?', '이목을 끄는 썸네일!', '광고 영상을 잘 만드려면?'])
if page == '홈':
    st.markdown(
    "*Handong Global University - Big Data Analysis 2023-01*"
    )
    # # 로고 이미지 로드
    # image = Image.open('logo.jpg')
    # st.image("logo.jpg", width=180)

    # 상단 제목
    st.subheader('🎥 자신에게 맞는 광고영상 특징을 모아보세요!')

    # 장르 선택 (한글로 표시)
    genre_kor = st.selectbox('장르를 선택하세요.', [None]+list(genre_dict.keys()))
    st.sidebar.markdown(f"현재 선택된 장르는 **{genre_kor}** 이에요.")
    
    # 장르 선택 리스트
    if genre_kor is not None:
        genre_eng = genre_dict[genre_kor]

        # 장르에 따른 데이터 필터링
        grouped_df = df.groupby('genre')
        genre_df = grouped_df.get_group(genre_eng)
        st.session_state['genre'] = (genre_kor, grouped_df.get_group(genre_eng))
        all_tags, all_tags_str = make_one_str(genre_df, 'tag')
        # 단어의 빈도수 계산
        word_freq = Counter(all_tags)
        # 상위 10개 단어만 선택
        top_5 = word_freq.most_common(10)
        keywords = [item[0] for item in top_5]
        keywords_str = ", ".join(keywords)
        st.markdown(
        f'<p style="color:orange;"><strong>💡 하이픈은 {genre_kor}에서 "{keywords_str}"의 태그를 발견했어요!</strong></p>',
        unsafe_allow_html=True,
        )
        st.success(emoticon_dict[genre_kor] +' '+ f"**왼쪽 메뉴에서 '{genre_kor}' 광고 영상의 특징을 골라보세요.**")

elif page == '어떤 제목이 인기가 많을까?':
    if st.session_state['genre'] is not None:
        with st.spinner('**영상들을 분석하고 있어요...**'):
            genre_kor, genre_df = st.session_state['genre']
            st.sidebar.markdown(f"현재 선택된 장르는 **{genre_kor}** 이에요.")
            st.title(emoticon_dict[genre_kor] + ' '+ f"'{genre_kor}' 장르는...")
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("🔤 영상 제목에 이러한 키워드가 많아요.")
            plot_wordcloud(genre_df, text_feature='title')
            
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader(f"✍️ 영상 제목에 자주 등장하는 키워드에요.")
            freq_words_df = plot_freq_keyword(genre_df, 'title_token')
            st.subheader("👇 키워드에 대한 점수를 확인해보세요!")
            st.info(f"**점수 = 영상조회수/채널평균조회수 의 평균**")
            st.write(freq_words_df)
        
elif page == '이목을 끄는 썸네일!':
    if st.session_state['genre'] is not None:
        with st.spinner('**영상들을 분석하고 있어요...**'):
            genre_kor, genre_df = st.session_state['genre']
            genre_eng = genre_dict[genre_kor]
            st.sidebar.markdown(f"현재 선택된 장르는 **{genre_kor}** 이에요.")
            st.title(emoticon_dict[genre_kor] + ' '+ f"'{genre_kor}' 장르는...")
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader(f"📝 썸네일에서 문자 영역이 {round(genre_df['thumbnail_text_ratio'].mean()*100)}% 를 차지해요.")
            st.info(f"**썸네일의 문자영역은 평균적으로 {round(df['thumbnail_text_ratio'].mean()*100)}% 에요.**")
            st.markdown("<hr>", unsafe_allow_html=True)
            ### 색깔 정보
            st.subheader(f"🌈 썸네일에서 색상, 명도, 채도를 살펴봐요.")
            st.subheader(f"🟠썸네일의 색상이 {color_print(color_df, genre_eng, 'color_category')}")
            st.subheader(f"🟡 썸네일의 명도가 {color_print(color_df, genre_eng, 'lightness_category')}")
            st.subheader(f"🟢 썸네일의 채도가 {color_print(color_df, genre_eng, 'saturation_category')}")
            st.info("**전체 채널의 사분위수 범위**")
            st.markdown("<hr>", unsafe_allow_html=True)
            
            st.subheader(f"✍️ 썸네일에 자주 등장하는 키워드에요.")
            freq_words_df = plot_freq_keyword(genre_df, 'thumbnail_text_token') 
            st.subheader("👇 키워드에 대한 점수를 확인해보세요!")
            st.info(f"**점수 = 영상조회수/채널평균조회수 의 평균**")
            st.write(freq_words_df)
            all_tags, all_tags_str = make_one_str(genre_df, 'thumbnail_labels_translate')
            
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("📷 썸네일에서 탐지된 객체들을 보여드릴게요.")
            wordcloud = WordCloud(width=1200, height=800, background_color='white', font_path='BMDOHYEON_ttf.ttf').generate(all_tags_str)
            plt.figure(figsize=(8, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
elif page == '광고 영상을 잘 만드려면?':
    if st.session_state['genre'] is not None:
        genre_kor, genre_df = st.session_state['genre']
        st.sidebar.markdown(f"현재 선택된 장르는 **{genre_kor}** 이에요.")
        duration_dist = genre_df['duration_min'].describe()
        mean_time = duration_dist['mean']
        genre_ratio = round((len(genre_df)/len(df)) * 100)
        st.title(emoticon_dict[genre_kor] + ' '+ f'{genre_kor} 장르는...')
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader(f"🔍 인기있는 유료광고 영상 중 {genre_ratio}% 를 차지하고 있어요.")
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("⏱️ 평균 영상 길이는 " + convert_time(mean_time)+ "에요.")
        st.info(f"**전체 영상 평균 길이는 {convert_time(df['duration_min'].mean())}에요.**")
        
        st.subheader(f"💯 영상들의 평균 점수는 {round(genre_df['score'].mean(),2)} 점이에요.")
        st.info(f"**전체 영상 평균 점수는 {round(df['score'].mean(),2)} 점이에요.**")
        st.info(f"**점수 = 영상조회수/채널평균조회수 의 평균**")


