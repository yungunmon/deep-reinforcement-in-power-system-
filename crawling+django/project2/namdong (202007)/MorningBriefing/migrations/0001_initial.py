# Generated by Django 2.1.5 on 2019-03-22 09:35

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='News',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Title', models.CharField(max_length=200)),
                ('Link', models.CharField(max_length=2000, unique=True)),
                ('Comment', models.TextField(blank=True, default=None)),
                ('Published_date', models.DateField(default=django.utils.timezone.now)),
                ('Display', models.CharField(choices=[('A', '보고서출력'), ('B', '보고서제거')], default='B', max_length=1)),
                ('Data_field', models.CharField(blank=True, choices=[('A', '발전사 동향'), ('B', '전력산업/시장'), ('C', '신재생에너지/기술'), ('D', '경제/에너지')], default='m', max_length=1)),
            ],
            options={
                'ordering': ['Data_field', 'Published_date'],
            },
        ),
        migrations.CreateModel(
            name='Report',
            fields=[
                ('idx', models.AutoField(primary_key=True, serialize=False)),
                ('site', models.CharField(max_length=50)),
                ('title', models.TextField(default=None, unique=True)),
                ('link', models.TextField(default=None)),
                ('published_date', models.DateField(default=None)),
            ],
        ),
    ]
