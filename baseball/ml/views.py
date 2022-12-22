
from django.shortcuts import render

import warnings
warnings.filterwarnings("ignore")

from ml.utils import *


def test(requset):
    return render(requset, 'ml/year_ops.html')


def index(request):
    return render(request, 'ml/index.html')


def predict_page(request):
    context = {
        'sample': 'sample'
    }
    return render(request, 'ml/predict_form.html', context)



def predict(request):
    batter_name = request.POST['batter_name']
    age = float(request.POST['age'])
    lag1_OBP = float(request.POST['lag1_OBP'])
    lag2_OBP = float(request.POST['lag2_OBP'])
    lag3_OBP = float(request.POST['lag3_OBP'])
    mean_OBP = float(request.POST['mean_OBP'])
    algorithm = request.POST['algorithm']

    df = pd.DataFrame([[age, lag1_OBP, lag2_OBP, lag3_OBP, mean_OBP]])

    obp, slg = predict_score(df, algorithm)
    score = np.append(obp, slg)
    score_chart = score_bar_chart(score)

    obp_chart = obp_impt_graph(df)
    slg_chart = slg_impt_graph(df)

    team_ops_chart = team_ops()
    context = {
        'batter_name' : batter_name,
        'OPS' : obp + slg,
        'OBP': obp,
        'SLG': slg,
        'score_chart': score_chart,
        'obp_chart':obp_chart,
        'slg_chart' :slg_chart,
        'team_ops_chart' : team_ops_chart
    }


    return render(request, 'ml/result.html', context)


def text_analyze(request):
    context = {}
    if request.GET.get('keyword'):
        keyword = request.GET['keyword']
        chart = get_tweets(keyword)
        # pos, neg, posCnt, negCnt, neuCnt = pos_neg()
        context = {
            'keyword': keyword,
            'chart': chart,
        }

    return render(request, 'ml/text_analyze.html', context)


def positive_negative_model(request):
    context = {}
    if request.GET.get('keyword'):
        keyword = request.GET['keyword']
        pos_chart, neg_chart, pos_cnt, neg_cnt, neu_cnt = positive_negtive_model(keyword)
        context = {
            'keyword': keyword,
            'pos_chart': pos_chart,
            'neg_chart' : neg_chart,
            'pos_cnt' : pos_cnt,
            'neg_cnt' : neg_cnt,
            'neu_cnt' : neu_cnt
        }
    return render(request, 'ml/pos_neg.html', context)

def about(request):
    return render(request, 'ml/about.html')