from django.urls import path
from .views import *

urlpatterns = [
     path('agent/test/', TestView.as_view(), name='test'),
     path('agent/chat/', AskQuestionView.as_view(), name='chat')
]
