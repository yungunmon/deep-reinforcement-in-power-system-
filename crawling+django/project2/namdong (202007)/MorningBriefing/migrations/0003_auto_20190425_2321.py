# Generated by Django 2.1.5 on 2019-04-25 14:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MorningBriefing', '0002_news_company'),
    ]

    operations = [
        migrations.AlterField(
            model_name='news',
            name='Data_field',
            field=models.CharField(blank=True, choices=[('A', '발전사 동향'), ('B', '전력산업/시장'), ('C', '신재생에너지/기술'), ('D', '에너지'), ('E', '에너지')], default='m', max_length=1),
        ),
    ]
