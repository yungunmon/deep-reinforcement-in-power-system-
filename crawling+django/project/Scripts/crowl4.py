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

def main():
    folder_name= 'svr'
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('--disable-gpu')
    options.add_argument('lang=ko_KR')
    path = 'C:/Users/svr/project/Scripts/chromedriver.exe'    
    driver = webdriver.Chrome(path)
    driver.get('http://koenergy.biz:8080/crawl')
    time.sleep(1000)
    print("good")
    
if __name__ == "__main__":
    main()

