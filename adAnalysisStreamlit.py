import streamlit as st
# Data Structure
import pandas as pd
import numpy as np
from ast import literal_eval
# Visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# Text Analysis
from wordcloud import WordCloud
from collections import Counter, defaultdict
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


# font_dir = 'https://raw.githubusercontent.com/Hasaero/Content-Evaluation-Model/master/BMDOHYEON_ttf.ttf'
# fm.fontManager.addfont(font_dir)
# fm._load_fontmanager(try_read_cache=False)
# plt.rc('font', family='BM Dohyeon')
url = 'https://raw.githubusercontent.com/Hasaero/Content-Evaluation-Model/master/BMDOHYEON_ttf.ttf'
response = requests.get(url)

# 현재 디렉토리에 ttf 파일을 저장합니다.
with open('BMDOHYEON_ttf.ttf', 'wb') as out_file:
    out_file.write(response.content)

# 이제 파일은 로컬 파일 시스템에 저장되어 있으므로 ft2font.FT2Font에서 사용할 수 있습니다.
font_path = os.path.abspath('BMDOHYEON_ttf.ttf')
fm.fontManager.addfont(font_path)

# 위 코드는 캐시된 FontManager를 무시하고 새로운 것을 불러오도록 설정합니다.
fm._load_fontmanager(try_read_cache=False)

# 이제 'BM Dohyeon' 폰트를 사용할 수 있게 됐습니다.
plt.rc('font', family='BM Dohyeon')
#os.chdir('C:\\Users\\7info\\Desktop\\Content_Evaluation')
# 데이터 로딩

def plot_wordcloud(df, text_feature, font_path=url):
    
    new_df = df.dropna(subset=[text_feature])
    text = ' '.join(new_df[text_feature])
    wordcloud = WordCloud(width=1200, height=800, background_color='white', font_path='BMDOHYEON_ttf.ttf').generate(text)
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



def plot_freq_keyword(df, text_feature):
    
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

def convert_to_list(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError):
        return None  # or whatever you want to return for invalid values
    
# df = pd.read_csv('good_ad_data.csv')
# color_df = pd.read_csv('good_ad_color.csv')
df = pd.read_csv('https://raw.githubusercontent.com/Hasaero/Content-Evaluation-Model/master/good_ad_data.csv')
df['title_token'] = df['title_token'].apply(convert_to_list)
df['thumbnail_text_token'] = df['thumbnail_text_token'].apply(convert_to_list)

color_df = pd.read_csv('https://raw.githubusercontent.com/Hasaero/Content-Evaluation-Model/master/good_ad_color.csv')

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


