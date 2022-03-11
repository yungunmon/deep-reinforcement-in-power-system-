
def main():

    from selenium import webdriver
    from bs4 import BeautifulSoup
    from selenium.webdriver.support.ui import Select

    
    import re
    import datetime
    import os
    import shutil
    import timedelta
    import threading
    import time
    folder_name= 'svr'
    #======== 셀 레 니 움 설 정 ========#
    options = webdriver.ChromeOptions()
    #헤들리스로 만들기-----------------
    #options.add_argument('headless')
    #options.add_argument('--disable-gpu')
    #options.add_argument('lang=ko_KR')
    #----------------------------------
    path = 'C:/Users/svr/project/Scripts/chromedriver.exe'
    #chrome_options=options
    driver = webdriver.Chrome(path)
    driver.get('http://koenergy.biz:8080/crawl')
    time.sleep(1000)

if __name__ == "__ main__":
    main()
