from django import forms
from .models import News
from .models import Report
class CommentForm(forms.ModelForm): #forms의 ModelForm 클래스를 상속 받는다.

    class Meta:
        model = News
        fields = ('Comment', )
class NewTitleForm(forms.ModelForm): #forms의 ModelForm 클래스를 상속 받는다.

    class Meta:
        model = News
        fields = ('Title',)