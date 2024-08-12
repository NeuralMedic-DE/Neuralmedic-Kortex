from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),
    path('download/', views.download_file, name='download_file'),
path('send-metadata/', views.send_metadata, name='send_metadata'),
]

from . import consumers
from django.urls import re_path
websocket_urlpatterns = [
    re_path(r'ws/eeg-data/$', consumers.EEGConsumer.as_asgi()),
]


if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)