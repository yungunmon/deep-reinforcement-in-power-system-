# Generated by Django 2.1.5 on 2019-04-25 12:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MorningBriefing', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='news',
            name='Company',
            field=models.CharField(default=True, max_length=200),
        ),
    ]
