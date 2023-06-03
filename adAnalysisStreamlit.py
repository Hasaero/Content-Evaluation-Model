import streamlit as st
# Data Structure
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Text Analysis
from wordcloud import WordCloud
from collections import Counter, defaultdict
from konlpy.tag import Okt

# Vision
from PIL import Image

# System
import io
import os
from io import BytesIO

# Access the internet
from urllib.request import urlopen
import requests

# 작업 경로 설정
os.chdir('C:\\Users\\7info\\Desktop\\Content_Evaluation')

# 데이터 로딩
df = pd.read_csv('good_ad_data.csv')

def plot_wordcloud(genre_df, text_column='title_token', font_path='NanumGothic'):
    plt.rc('font', family='Malgun Gothic')
    
    text = ' '.join(genre_df[text_column])
    wordcloud = WordCloud(width=1200, height=800, background_color='white', font_path=font_path).generate(text)
    plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{genre_df["genre"].unique()[0]}', fontsize=20)
    st.pyplot(plt)

# 한글화를 위한 장르 딕셔너리
genre_dict = {
    'Pets & Animals': '동물',
    'Autos & Vehicles': '자동차',
    'People & Blogs': '일상',
    'Howto & Style': '방법 & 스타일',
    'Travel & Events': '여행',
    'Music': '음악',
    'Gaming': '게임',
    'Education': '교육',
    'Science & Technology': '과학 & 기술',
    'Entertainment': '엔터테인먼트',
    'Comedy': '코미디',
    'Sports': '스포츠',
    'News & Politics': '뉴스 & 정치',
    'Film & Animation': '영화 & 애니메이션'
}

# 이미지 로드
image1 = Image.open('ugwangi.png')
image2 = Image.open('handong.jpg')

# 이미지 출력 (좌측 상단)
cols = st.beta_columns(3)

cols[0].image(image1, use_column_width=True)
cols[1].write('X', align='center')
cols[2].image(image2, use_column_width=True)

# 타이틀 설정
st.title('인기있는 유료광고 영상 특징 분석')
st.write('자신에게 맞는 광고영상 특징을 모아보세요.')

# '시작하기' 버튼 설정
start_button = st.button('시작하기')

# 버튼 클릭 시 장르 선택 및 워드 클라우드 출력
if start_button:
    # 장르 선택 (한글로 표시)
    selected_genre_eng = st.selectbox('장르를 선택하세요.', list(genre_dict.keys()))
    selected_genre_kor = genre_dict[selected_genre_eng]

    # 장르에 따른 데이터 필터링
    grouped_df = df.groupby('genre')
    genre_df = grouped_df.get_group(selected_genre_eng)

    # 데이터프레임 출력
    st.write(genre_df)

    plot_wordcloud(genre_df, text_column='title_token', font_path='NanumGothic')
