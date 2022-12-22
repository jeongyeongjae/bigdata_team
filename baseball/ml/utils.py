import base64
from nltk import Text
import re
import networkx as nx
from .models import *
from io import BytesIO

from matplotlib import font_manager, rc
import platform
import pickle
import tweepy
from konlpy.tag import Okt

from collections import Counter
from apyori import apriori

import matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

CONSUMER_KEY = "IwTf0j3k2Yy3NXlcc4cRDDmJ1"
CONSUMER_SECRET = "AgGCcabil35IvI3mJ9BhIfiyEEhgrNNVcWmI9EpX6CLUPvBfnz"
ACCESS_TOKEN_KEY = "1322415736975863809-71Rl1kqD7ZJtX2YJA6aZkKOaig4d3S"
ACCESS_TOKEN_SECRET = "VHnsQjMRyHyXKEHd1wMwBkKXzxBEFOw87JyHL8H7Wz2N8"


##################### CHART ##########################

def chart_init():
    if platform.system() == 'Windows':
        # 윈도우인 경우 맑은 고딕 폰트 이용
        font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf"
                                                ).get_name()
        rc('font', family=font_name)
    else:
        # Mac 인 경우
        rc('font', family='AppleGothic')

    # 그래프에서 마이너스 기호가 표시되게
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.switch_backend('AGG')


def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def score_bar_chart(score):
    chart_init()

    fig = plt.figure(figsize=(5, 4))
    x_label = ['OBP', 'SLG']

    plt.bar(x_label, score)

    plt.title('OBP와 SLG 값', fontsize=20)
    plt.xlabel('', fontsize=18)
    plt.ylabel('', fontsize=18)

    plt.ylim(0, 0.5)

    for i, v in enumerate(score):
        plt.text(i - 0.1, v + 0.01, str(np.round(v, 3)))

    plt.tight_layout()
    chart = get_graph()

    return chart



def team_ops():
    chart_init()
    regular_season_df = pd.read_csv("/Users/jueon/Downloads/62540_KBO_prediction_data/Regular_Season_Batter.csv")
    med_OPS_team = regular_season_df.pivot_table(index=['team'], columns='year',
                                                 values='OPS', aggfunc='median')
    # 2005년 이후에 결측치가 존재하지 않는 팀만 확인
    team_idx = med_OPS_team.loc[:, 2005:].isna().sum(axis=1) <= 0


    plt.plot(med_OPS_team.loc[team_idx, 2005:].T, marker='o', markersize=4)
    plt.grid(axis='y', linestyle='-', alpha=0.4)
    plt.legend(med_OPS_team.loc[team_idx, 2005:].T.columns,
               loc='center left', bbox_to_anchor=(1, 0.5))  # 그래프 범례를 그래프 밖에 위치
    plt.title('연도별 팀 OPS')
    plt.tight_layout()
    chart = get_graph()

    return chart



def wrmse_chart(wrmse_score):
    plt.switch_backend('AGG')

    x_lab = ['Lasso', 'Ridge', 'RF', 'XGB']

    plt.bar(x_lab, wrmse_score)
    plt.title('WRMSE of OBP', fontsize=20)
    plt.xlabel('model', fontsize=18)
    plt.ylabel('', fontsize=18)
    plt.ylim(0, 0.5)

    # 막대그래프 위에 값을 표시해준다.
    for i, v in enumerate(wrmse_score):
        plt.text(i - 0.1, v + 0.01, str(np.round(v, 3)))  # x 좌표, y 좌표, 텍스트를 표현한다.

    plt.tight_layout()
    chart = get_graph()

    return chart


##################################################


def wrmse(v, w, p):
    # v: 실제값
    # w: 타수
    # p: 예측값
    return sum(np.sqrt(((v - p) ** 2 * w) / sum(w)))




def predict_score(DataFrame, algorithm):
    obp_path = "ml/MLmodels/" + algorithm + "_OBP_model.sav"
    slg_path = "ml/MLmodels/" + algorithm + "_SLG_model.sav"

    obp_model = pickle.load(open(obp_path, "rb"))
    slg_model = pickle.load(open(slg_path, "rb"))

    obp = obp_model.predict(DataFrame)
    slg = slg_model.predict(DataFrame)

    return obp, slg


def score_bar_chart(score):
    chart_init()

    fig = plt.figure(figsize=(5, 5))
    x_label = ['OBP', 'SLG']

    plt.bar(x_label, score)

    plt.title('OBP와 SLG 값', fontsize=20)
    plt.xlabel('', fontsize=18)
    plt.ylabel('', fontsize=18)

    plt.ylim(0, 0.5)

    for i, v in enumerate(score):
        plt.text(i - 0.1, v + 0.01, str(np.round(v, 3)))

    plt.tight_layout()
    chart = get_graph()

    return chart


def slg_impt_graph(dataFrame):
    chart_init()
    y_vars = ['age', '1년전 OBP', '2년전 OBP','3년전 OBP', '평균 OBP']

    slg_path = "ml/MLmodels/RF_SLG_model.sav"


    slg_model = pickle.load(open(slg_path, "rb"))

    fig = plt.figure(figsize=(10, 8))


    plt.barh(y_vars, slg_model.feature_importances_)
    plt.title('변수별 SLG 성적 예측시의 중요도')

    plt.tight_layout()
    graph = get_graph()

    return graph


def obp_impt_graph(dataFrame):
    chart_init()
    y_vars = ['age', '1년전 OBP', '2년전 OBP','3년전 OBP', '평균 OBP']
    obp_path = "ml/MLmodels/RF_OBP_model.sav"


    obp_model = pickle.load(open(obp_path, "rb"))

    fig = plt.figure(figsize=(10, 8))

    plt.barh(y_vars, obp_model.feature_importances_)
    plt.title('변수별 OBP 성적 예측시의 중요도')
    plt.tight_layout()
    graph = get_graph()

    return graph


# 시간 변수를 생성하는 함수 정의
# AB, year, OBP, name 필요함
def lag_function(df, var_name, past):
    # df = 시간변수를 생성할 데이터 프레임
    # var_name = 시간변수 생성의 대상이 되는 변수 이름
    # past = 몇 년 전의 성적을 생성할지 결정 (정수형)
    df.reset_index(drop=True, inplace=True)

    # 시간변수 생성
    df['lag' + str(past) + '_' + var_name] = np.nan;
    df['lag' + str(past) + '_' + 'AB'] = np.nan

    for col in ['AB', var_name]:
        for i in range(0, (max(df.index) + 1)):
            val = df.loc[(df['batter_name'] == df['batter_name'][i]) &
                         (df['year'] == df['year'][i] - past), col]
            # 과거 기록이 결측치가 아니라면 값을 넣기
            if (len(val) != 0):
                df.loc[i, 'lag' + str(past) + '_' + col] = val.iloc[0]

    # 30타수 미만 결측치 처리
    df.loc[df['lag' + str(past) + '_' + 'AB'] < 30,
           'lag' + str(past) + '_' + var_name] = np.nan
    df.drop('lag' + str(past) + '_' + 'AB', axis=1, inplace=True)

    return df


def get_tweets(keyword):
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    tweets = api.search_tweets(keyword)

    # 크롤링된 데이터를 저장할 데이터 프레임입니다.
    columns = ['created', 'tweet_text']
    df = pd.DataFrame(columns=columns)

    # 크롤링된 데이터를 저장할 데이터 프레임입니다.
    columns = ['created', 'tweet_text']
    df = pd.DataFrame(columns=columns)

    # 크롤링을 수행할 갯수를 입력하고, Cursor 객체를 사용하여 크롤링을 수행합니다.
    max_tweets = 1000
    searched_tweets = [status for status in tweepy.Cursor(api.search_tweets, q=keyword).items(max_tweets)]

    for tweet in searched_tweets:
        tweet_json = tweet._json
        tweet_text = tweet_json['text']
        created = tweet_json['created_at']
        row = [created, tweet_text]
        series = pd.Series(row, index=df.columns)
        df = df.append(series, ignore_index=True)

    df.to_csv("ml/static/tweet_temp.csv", index=False)

    df = pd.read_csv("ml/static/tweet_temp.csv")

    df['ko_text'] = df['tweet_text'].apply(lambda x: text_cleaning(x))
    df['nouns'] = df['ko_text'].apply(lambda x: get_nouns(x))

    transactions = df['nouns'].tolist()
    transactions = [transaction for transaction in transactions if transaction]  # 공백 문자열을 방지합니다.

    # 연관 분석 수행
    results = list(apriori(transactions,
                           min_support=0.05,
                           min_confidence=0.1,
                           min_lift=5,
                           max_length=2))

    # 데이터 프레임 형태로 정리
    columns = ['source', 'target', 'support']
    network_df = pd.DataFrame(columns=columns)

    # 규칙의 조건절을 source, 결과절을 target, 지지도를 support 라는 데이터 프레임의 피처로 변환합니다.
    for result in results:
        if len(result.items) == 2:
            items = [x for x in result.items]
            row = [items[0], items[1], result.support]
            series = pd.Series(row, index=network_df.columns)
            network_df = network_df.append(series, ignore_index=True)


    # 말뭉치 추출
    tweet_corpus = "".join(df['ko_text'].tolist())

    # 명사 키워드 추출
    nouns_tagger = Okt()
    nouns = nouns_tagger.nouns(tweet_corpus)
    count = Counter(nouns)

    # 한글자 키워드를 제거합니다.
    remove_char_counter = Counter({x: count[x] for x in count if len(x) > 1})

    # 단어 빈도 점수 추가
    # 키워드와 키워드 빈도 점수를 node, nodesize 라는 데이터 프레임으로
    node_df = pd.DataFrame(remove_char_counter.items(), columns=['node', 'nodesize'])
    node_df = node_df[node_df['nodesize'] >= 50]  # 시각화의 편의를 위해 ‘nodesize’ 50 이하는 제거합니다.


    chart_init()
    plt.figure(figsize=(15, 15))

    # networkx 그래프 객체를 생성합니다.
    G = nx.Graph()

    # node_df의 키워드 빈도수를 데이터로 하여, 네트워크 그래프의 ‘노드’ 역할을 하는 원을 생성합니다.
    for index, row in node_df.iterrows():
        G.add_node(row['node'], nodesize=row['nodesize'])

    # network_df의 연관 분석 데이터를 기반으로, 네트워크 그래프의 ‘관계’ 역할을 하는 선을 생성합니다.
    for index, row in network_df.iterrows():
        G.add_weighted_edges_from([(row['source'], row['target'], row['support'])])

    # 그래프 디자인과 관련된 파라미터를 설정합니다.
    pos = nx.spring_layout(G, k=0.6, iterations=50)
    sizes = [G.nodes[node]['nodesize'] * 25 for node in G]
    nx.draw(G, pos=pos, node_size=sizes)

    # Windows 사용자는 AppleGothic 대신,'Malgun Gothic'. 그 외 OS는 OS에서 한글을 지원하는 기본 폰트를 입력합니다.
    nx.draw_networkx_labels(G, pos=pos, font_family='AppleGothic', font_size=25)

    # 그래프를 출력합니다.
    ax = plt.gca()
    plt.tight_layout()
    chart = get_graph()

    return chart


# 텍스트 정제 함수 : 한글 이외의 문자는 전부 제거합니다.
def text_cleaning(text):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')  # 한글의 정규표현식을 나타냅니다.
    result = hangul.sub('', text)
    return result


def get_nouns(x):
    korean_stopwords_path = "ml/static/korean_stopwords.txt"
    with open(korean_stopwords_path, encoding='utf8') as f:
        stopwords = f.readlines()
    stopwords = [x.strip() for x in stopwords]

    nouns_tagger = Okt()
    nouns = nouns_tagger.nouns(x)

    # 한글자 키워드를 제거합니다.
    nouns = [noun for noun in nouns if len(noun) > 1]

    # 불용어를 제거합니다.
    nouns = [noun for noun in nouns if noun not in stopwords]

    return nouns



def positive_negtive_model(keyword):
    tweet = np.load("ml/static/tweet_mat.npy")
    tweet01 = pd.read_csv("ml/static/tweet_temp.csv")
    # 불용어목록
    stopwords = ("ml/static/korean_stopwords.txt")
    tweet_model = load_model("ml/static/model.h5", compile=False)

    tweet_use = tweet_model.predict(tweet)

    # 라벨 예측
    tweet_label = np.argmax(tweet_use, axis=1)

    count01 = 0  # 긍정 개수
    count02 = 0  # 부정 개수
    count03 = 0  # 중립 개수
    lis01 = []  # 긍정 리스트
    lis02 = []  # 부정 리스트
    lis03 = []  # 중립 리스트

    for i in range(len(tweet01)):
        if tweet_label[i] - 1 == 1:  # 긍정시 개수 +1
            count01 += 1
            lis01.append(tweet01['tweet_text'].iloc[i])  # 긍정리스트에 추가
        elif tweet_label[i] - 1 == -1:
            count02 += 1
            lis02.append(tweet01['tweet_text'].iloc[i])
        else:
            count03 += 1
            lis03.append(tweet01['tweet_text'].iloc[i])

    result01 = lis01
    result02 = lis02
    result03 = lis03

    okt = Okt()
    # 형태소 분석, 그래프 생성
    for sentence in result01:
        temp01 = []
        temp01 = okt.morphs(sentence, stem=True)  # 토큰화
    test01 = Text(okt.morphs(sentence, stem=True), name="test")

    chart_init()
    plt.title('긍정')
    plt.plot(test01)
    plt.tight_layout()
    pos_chart = get_graph()

    for sentence in result02:
        temp02 = []
        temp02 = okt.morphs(sentence, stem=True)  # 토큰화
    test02 = Text(okt.morphs(sentence, stem=True), name="test")

    chart_init()
    plt.title('부정')
    plt.plot(test02)
    plt.tight_layout()
    neg_chart = get_graph()

    plt.show()

    return pos_chart, neg_chart, count01, count02, count03

