from django.urls import path

from ml import views

urlpatterns = [
    path('', views.index, name='index'),
    # path('graph/', views.get_chart_year_ops, name='year_ops'),
    path('predict_form_page/', views.predict_page, name='predict_page'),
    path('predict/', views.predict, name='predict'),
    path('test/', views.test, name='test'),
    path('text-analyze', views.text_analyze, name='text_analyze'),
    path('pos_neg', views.positive_negative_model, name='pos_neg'),
    path('about', views.about, name='about')
]
