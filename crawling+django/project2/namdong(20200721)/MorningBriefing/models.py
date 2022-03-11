from django.db import models
import datetime
from django.utils import timezone



class News(models.Model):
    Title = models.CharField(max_length=200) #<title>{% News.title%}</title>
    Link = models.CharField(max_length=2000, unique=True) #a href = {% News.link[]%}
    Published_date = models.DateField(default= None)
    Company = models.CharField(max_length=200 ,default= None)
    Importance = models.CharField(max_length=20 ,default= 0)
    Report_display = (
        ('A', '보고서출력'),
        ('B', '보고서제거'),
    )
    Display = models.CharField(
        max_length = 1,
        choices = Report_display,
        default = 'B',
    )

    Select = (
        ('A', '발전사 동향'), 
        ('B', '전력산업/시장'), 
        ('C', '신재생에너지/기술'),
        ('D', '경제'),
        ('E', '에너지'),
    )
    Data_field = models.CharField(
        max_length = 1,
        choices = Select,
        blank = True,
        default = 'm',
    )

    class Meta:
        ordering = ['Data_field', 'Published_date' , 'Importance'  ]
    def __str__(self):
        return self.Title

class Comment(models.Model):
    news =models.ForeignKey(News, default=None, on_delete=models.CASCADE, null=True)
    content = models.TextField(default=None)
    
class Report(models.Model):
    idx = models.AutoField(primary_key=True)
    site = models.CharField(max_length=50)
    title = models.TextField(default = None,unique=True)
    link = models.TextField(default = None)
    published_date = models.DateField(default = None)
    def __str__(self):
            return self.site


