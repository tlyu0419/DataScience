# Concurrent computing

## How to choose?
- 什麼是 CPU密集型計算(CPU bound)、I/O密集型計算(I/O bound)
  > Bound 指的是程式的運行最終會受到 CPU的限制還是I/O的限制
- CPU密集型計算
  - CPU密集型也叫做計算密集型，是指 I/O 在很短的時間就可以完成，CPU需大量的計算和處理，特點是 CPU 佔用率相當高。例如壓縮/解壓縮，加密/解密、正則表達式搜索
- I/O密集型計算
  - I/O密集型指的是系統運作大部分的情況是CPU在等I/O(硬碟/記憶體)的讀/寫操作，CPU佔用率仍然較低。例如文件處理程式，網路爬蟲程式、讀寫資料庫程式
  
## 多線程、多進程、多協程的對比
  - 一個進程中可以啟動多個線程，一個線程中可以啟動多個協程，彼此是包含關係
  - 從資源消耗的角度來看，由大到小依序為 進程 > 線程 > 協程
  
### Multiprocess(多進程)
- 優點: 可以利用多核 CPU 並行運算
- 缺點: 佔用資源最多，可啟動數目比線程少
- 適用於: CPU密集型計算
- Python package: multiprocessing

### Multithread(多線程)
- 優點: 相比進程較輕量、佔用較少資源
- 缺點:
  - 相比進程: 多線程只能並發執行，不能利用多CPU(GIL)
  - 相比協程: 能啟動的數量較少，較佔用記憶體
- 適用於: I/O 密集型計算，同時運行的任務數目要求不多
- Python package(threading)

### Multicoroutine(多協程)
- 優點: 記憶體消耗最少、啟動協程的數量最多
- 缺點: 支持的套件有限制，如不支持 request(但可用 aiohttp 取代)、程式開發較複雜
- 適用於: I/O 密集型計算，需要超多任務運行，但有現成套件支持的場景
- Python package: asyncio
  
## 怎麼根據任務選擇對應技術?
1. 確認任務的性質為 CPU密集型 還是 I/O密集型，如是 CPU 密集型就使用 multiprocessing
2. 如是I/O密集型
   - 接著確認以下3個問題:
     - 需要超多任務量?
     - 有現成協程套件支持?
     - 協程實現複雜度可以接受?(開發成本)
    - 如否，選擇使用多線程
    - 如是，選擇使用多協程

## 多線程數據通信的 queue.Queue
- queue.Queue可以用於多線程之間的線程安全的數據通信
  ```python
  # 帶入套件
  import queue

  # 創建Queue
  q = queue.Queue()

  # 添加元素
  q.put(item)

  # 獲取元素
  item = q.get()

  # 查詢狀態

  ## 查看元素的數量
  q.qsize()

  ## 判斷是否為空
  q.empty()

  ## 判斷是否已滿
  q.full()
  ```

## Sample
```python
def withdraw(account, amount):
    if account.balance >= amount:
        account.balance -= amount
        print(f'提領成功: 你的帳戶餘額為 {account.balance}')
    else:
        print(f'提領失敗: 你的帳戶僅剩 {account.balance}')
```

## Ref
- [【Python并发编程】怎样选择多线程多进程多协程](https://www.zhihu.com/zvideo/1304576820973662208)
- [Python速度慢的罪魁祸首，全局解释器锁GIL](https://www.zhihu.com/zvideo/1305941095898333184)
- [使用多线程，Python爬虫被加速10倍](https://www.zhihu.com/zvideo/1308557398148263936)
  - 多線程比單線程快了 16 倍
- [Python实现生产者消费者爬虫](https://www.zhihu.com/zvideo/1309297908651601920)