'''
Crawl 104 job bank data using multi-threading to speed up.
by tlyu0419
'''

import requests
import pandas as pd
import datetime
import json
import re
import os
import queue
import threading
import numpy as np
from tqdm import tqdm

# 地區代號
def get_areacode():
    url = 'https://static.104.com.tw/category-tool/json/Area.json'
    areacode = requests.get(url).text
    areacode = json.loads(re.sub('\ufeff','',areacode))[0]['n']
    areacode = pd.DataFrame(areacode)
    areacode = areacode.explode('n')
    areacode['des2'] = areacode['n'].apply(lambda x: x['des'])
    areacode['no2'] = areacode['n'].apply(lambda x: x['no'])
    areacode = areacode.loc[:,['des', 'no', 'des2', 'no2', 'eng']]
    areacode['no3'] = np.select(condlist=[areacode['des'].str.contains('台北|新北|桃園|台中|台南|高雄')], choicelist=[areacode['no2']], default=areacode['no'])
    return areacode['no3'].unique()

# 職類代號
def get_catecode():
    url = 'https://static.104.com.tw/category-tool/json/JobCat.json'
    catcode = re.sub('\ufeff', '', requests.get(url).text)
    catcode = pd.DataFrame(json.loads(catcode))
    catcode = catcode.explode('n')
    catcode['des2'] = catcode['n'].apply(lambda x: x['des'])
    catcode['no2'] = catcode['n'].apply(lambda x: x['no'])
    catcode['n2'] = catcode['n'].apply(lambda x: x['n'])
    catcode = catcode.explode('n2')
    catcode['des3'] = catcode['n2'].apply(lambda x: x['des'])
    catcode['no3'] = catcode['n2'].apply(lambda x: x['no'])
    catcode = catcode.loc[:,['des', 'no', 'des2', 'no2', 'des3', 'no3']]
    return catcode['no2'].unique()

# 產業代號
def get_indcate():
   url = 'https://static.104.com.tw/category-tool/json/Indust.json'
   resp = requests.get(url)
   catecode = pd.DataFrame(resp.json())
   catecode = catecode.explode('n') 
   catecode['des2'] = catecode['n'].apply(lambda x: x['des'])
   catecode['no2'] = catecode['n'].apply(lambda x: x['no'])
   catecode['n2'] = catecode['n'].apply(lambda x: x['n'])
   catecode = catecode.explode('n2')
   catecode['des3'] = catecode['n2'].apply(lambda x: x['des'])
   catecode['no3'] = catecode['n2'].apply(lambda x: x['no'])
   catecode = catecode.loc[:,['des', 'no', 'des2', 'no2', 'des3', 'no3']]
   return catecode['no2'].unique()

def get_jobabs(area, cate, workerid):
    data = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0',
           'Referer': 'https://www.104.com.tw/',
           }
    page = 1
    params = {'ro': 1, # 全職
              'isnew': 14, # 近14天內有更新
              'order': 16, # 按照更新日期欄位排序
              'asc': 0, # 新排到舊
              's9':1, # 上班時段 為日班
              # 'jobcat': cate,
              'mode': 'l',
              'indcat': cate, # 產業類型
              'area': area,
              'page': page}
    
    while True:
        try:
            url = 'https://www.104.com.tw/jobs/search/list'
            resp = requests.get(url, headers=headers, params=params)
            print(workerid, ':', resp.url)
            ndf = pd.DataFrame(resp.json()['data']['list'])
            if ndf.shape[0]==0:
                break
            data.append(ndf)

            if resp.json()['data']['pageNo'] == resp.json()['data']['totalPage']:
               break
            elif params['page'] == 100:
               break
            else:
                params['page'] =  params['page']+1
        except:
            print('==================== Error and retry ====================')
    if len(data) >= 1:
        df = pd.concat(data, ignore_index=True)
        df['job_link'] = df['link'].apply(lambda x: x['job'])
        df  = df[['jobNo', 'coIndustryDesc', 'custName', 'jobName', 'salaryDesc', 'landmark', 'major', 'mrtDesc', 'job_link']]
        df.to_parquet(f'./data/{area}_{cate}.parquet')
    # return data


class Worker(threading.Thread):
  def __init__(self, queue, workerid):
    threading.Thread.__init__(self)
    self.queue = queue
    self.workerid = workerid

  def run(self):
    while self.queue.qsize() > 0:
      # 取得新的資料
      area, cate = self.queue.get()

      # 處理資料
      get_jobabs(area, cate, self.workerid)

# Concat files under data folder
def concat_files():
    files = os.listdir('data')
    df = []
    for f in tqdm(files):
        ndf = pd.read_parquet(f'data/{f}')
        df.append(ndf)
    df = pd.concat(df, ignore_index=True)
    return df

if __name__ == '__main__':

    os.makedirs('data', exist_ok=True)
    # Unit test    
    # area = '6001001001'
    # cate = '1005000000'
    # get_jobabs(area=area, cate=cate, workerid=1)


    areacode = get_areacode()
    indcate = get_indcate()
    
    # 多線程加速
    my_queue = queue.Queue()
    
    # 將資料放入佇列
    for area in areacode:
       for cate in indcate:
          my_queue.put((area, cate))

    dt1 = datetime.datetime.now()

    # 建立 5 個 Worker
    my_worker1 = Worker(my_queue, 1)
    my_worker2 = Worker(my_queue, 2)
    my_worker3 = Worker(my_queue, 3)

    # 讓 Worker 開始處理資料
    my_worker1.start()
    my_worker2.start()
    my_worker3.start()

    # 等待所有 Worker 結束
    my_worker1.join()
    my_worker2.join()
    my_worker3.join()

    print("Done.")
    dt2 = datetime.datetime.now()


    print(dt2-dt1)