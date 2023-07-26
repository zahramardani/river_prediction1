from django.contrib import admin
from django.urls import path
from prediction.views import allsteps_preprocessing_data_split, get_combined_dataset, prediction_form, allsteps_preprocessing_data_before, allsteps_preprocessing_data_after

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', prediction_form, name='prediction_form'),
    path('get_combined_dataset/', get_combined_dataset, name='get_combined_dataset'),
    path('allsteps_preprocessing_data_before/', allsteps_preprocessing_data_before, name='allsteps_preprocessing_data_before'),
    path('allsteps_preprocessing_data_after/', allsteps_preprocessing_data_after, name='allsteps_preprocessing_data_after'),
    path('allsteps_preprocessing_data_split/', allsteps_preprocessing_data_split, name='allsteps_preprocessing_data_split'),
]
