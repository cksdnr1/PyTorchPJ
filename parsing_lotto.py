import warnings
import requests
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup


# wanring 메시지 출력 안함
warnings.filterwarnings('ignore')

# 최신 회차 크롤링 함수
def get_max_count():
    url = 'https://dhlottery.co.kr/common.do?method=main'
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'lxml')
    print(soup.find('strong', id='lottoDrwNo'))
    max_count = int(soup.find('strong', id='lottoDrwNo').text)
    return max_count

# 로또 당첨번호 정보 조회 함수
def crawling_lotto(count):
    # url에 회차를 실어 페이지 조회
    url = f'https://dhlottery.co.kr/gameResult.do?method=byWin&drwNo={count}'
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'lxml')
    
    date = datetime.strptime(soup.find('p', class_='desc').text, '(%Y년 %m월 %d일 추첨)')
    win_number = [int(i) for i in soup.find('div', class_='num win').find('p').text.strip().split('\n')]
    bonus_number = int(soup.find('div', class_='num bonus').find('p').text.strip())
    
    return {
        'date': date, 
        'win_number': win_number, 
        'bonus_number': bonus_number
    }

# 최신 회차 가져오기
max_count = get_max_count()
# 전체 회차 크롤링
data = pd.DataFrame()
for i in tqdm(range(1, max_count+1)):
    result = crawling_lotto(i)
    data = pd.concat([data, pd.DataFrame({'date': result['date'],
                        'num1': result['win_number'][0],
                        'num2': result['win_number'][1],
                        'num3': result['win_number'][2],
                        'num4': result['win_number'][3],
                        'num5': result['win_number'][4],
                        'num6': result['win_number'][5],
                        'bonus': result['bonus_number'],
                       },index=[0])], ignore_index=True)

data.to_csv('lotto-1052.csv', index=False)