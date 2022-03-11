from __future__ import absolute_import 
import os 
from celery import Celery 
from celery.schedules import crontab 
from django.conf import settings # noqa 

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'MorningBriefing.settings') 
app = Celery('MorningBriefing')
app.config_from_object('django.conf:settings', namespace = 'CELERY') 
app.autodiscover_tasks() 

# task setting 
@app.task(bind=True) 
def debug_task(self):
	print('Request: {0!r}'.format(self.request)) 

app.conf.beat_schedule = {'add-every-30-minutes-contrab':

 { 'task': 'check_new_update_ted', 'schedule': crontab(minute='*/30'), }, }
