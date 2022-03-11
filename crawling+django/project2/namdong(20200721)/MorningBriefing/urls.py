from django.conf.urls import url
from MorningBriefing import views
def deletePolls(request):
    pollId = deletePool(request.GET['id'])
    return HttpResponseRedirect("/page_path/")
app_name = 'MorningBriefing'
urlpatterns = [
	url(r'^$',views.index, name = 'home'),
	url(r'^발전사동향/',views.a, name='발전사동향'),
	url(r'^파워인사이트/',views.Powerinsight, name='파워인사이트'),
	url(r'^전력산업/시장/',views.b, name='전력산업/시장'),
	url(r'^신재생에너지/기술/',views.c, name='신재생에너지/기술'),
	url(r'^경제/',views.d ,name='경제'),
	url(r'^(?P<pk>\d+)/add_comment_to_post/$',views.add_comment_to_post, name='add_comment_to_post'),
	url(r'^(?P<pk>\d+)/delete_news/$',views.delete_news,name='delete_news'),
	url(r'^(?P<pk>\d+)/delete/$',views.delete,name='delete'),
	url(r'^(?P<pk>\d+)/Display_on/$',views.Display_on,name='Display_on'),
	url(r'^(?P<pk>\d+)/Display_off/$',views.Display_off,name='Display_off'),
	url(r'^보고서/',views.report ,name='보고서'),
	url(r'^보고서초기화/',views.reportinitialize ,name='보고서초기화'),
	url(r'^(?P<pk>\d+)/modify_title/$',views.modify_title, name='modify_title'),
	url(r'^crawl/',views.crawl, name='crawl'),
	url(r'^power/',views.power, name='power' ),
	url(r'^news_display/',views.news_display, name='news_display'),
	url(r'^news_expired/',views.news_expired,name='news_expired'),
	url(r'^add_comment/$',views.add_comment,name='add_comment'),
	url(r'^manage_news/$',views.manage_news,name='manage_news'),
	url(r'^manage_comments/$',views.manage_comments,name='manage_comments'),
	url(r'^뉴스수정/$',views.e,name='뉴스수정'),
]