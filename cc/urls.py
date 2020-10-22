from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('run_cc/<int:id_formation>/', views.run_control, name='run_control'),
    path('train/<int:id_formation>/', views.train, name='train'),
    path('Etudiants_list/<int:id_formation>/', views.get_etudiants_list, name='get_etudiants_list'),
    path('absances_list/', views.absances_list, name='absances_list'),
    path('Formations_list/', views.formations_list, name='formations_list'),
]
