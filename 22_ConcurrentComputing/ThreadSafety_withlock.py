import threading

lock = threading.Lock()

class Account:
    def __init__(self, balance):
        self.balance = balance

def withdraw(account, amount):
    with lock:
        if account.balance >= amount:
            print(f"{threading.current_thread().name}: 提領成功")
            print(f"{threading.current_thread().name}: 出鈔中，請稍後")
            account.balance -= amount
            print(f"{threading.current_thread().name}: 你的餘額為 {account.balance} 元")
        else:
            print(f"{threading.current_thread().name}: 餘額不足，你的帳戶僅剩 {account.balance}元")

if __name__ == "__main__":
    account = Account(1000)
    ta = threading.Thread(name="ta", target=withdraw, args=(account, 800))
    tb = threading.Thread(name="tb", target=withdraw, args=(account, 800))

    ta.start()
    tb.start()