# Generated by Django 2.2.4 on 2020-07-05 15:33

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('MorningBriefing', '0006_auto_20200705_1635'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='news',
            name='Comment',
        ),
        migrations.AddField(
            model_name='comment',
            name='news',
            field=models.ForeignKey(default=None, null=True, on_delete=django.db.models.deletion.CASCADE, to='MorningBriefing.News'),
        ),
    ]
