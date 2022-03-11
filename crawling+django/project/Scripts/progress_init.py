import schedule
import time
from selenium import webdriver
from bs4 import BeautifulSoup
import re
import datetime
import os
import shutil
import timedelta
import threading
import time

from selenium.webdriver.support.ui import Select

def job():
    folder_name= 'svr'
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('--disable-gpu')
    options.add_argument('lang=ko_KR')
    path = 'C:/Users/svr/project/Scripts/chromedriver.exe'    
    driver = webdriver.Chrome(path)
    driver.get('http://koenergy.biz:7890/progress_init')

schedule.every().hour.do(job)    


while True:
    schedule.run_pending()
    time.sleep(1)
