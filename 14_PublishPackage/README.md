# Three crucial points of the open-source project.

- What's your content?
- Support ways
- Package your project

## What's your content?
- Think about **why and what you must to do this project**, and it's the most vital part of all. 
    - Note: It's a good practice to google your idea to see there is a project identical to your's before you develop it!
        - [facebook_crawler](https://pypi.org/project/facebook-crawler/) v.s. [facebook-scraper](https://pypi.org/project/facebook-scraper/)
- Three good reasons to develop a new project
  - **New skill**: An innovative skill is always a good topic, such as how to improve the supervised/unsupervised models more accurate, or increase the performance of the program. 
  - **New application**: Apply an existing technic to new realm, such as analyze the job descriptions in human resources through NLP. Or analyze account transfer behavior through SNA.
  - **Good for society**: Attempt to resolve a specific social problem, such as fake news. 

- In my experience, the first one, New skill, may be more appropriate to someone who is a statistics/data engineer background. But the last two, **new application and Good for society, are the advantages of someone who is social science background**.
    - Moreover, I suppose that the last two topics are more acceptable for the conference. You can try this if you want to unlock the speaker's achievement.

## Support ways
- A short story
  - Many social enterprises would apply for subsidies from the government because they aim to do something good for society. 
  - **But what if the economy is in depression?** The government will decrease the subsidy budget and directly induce them to be closed or bankrupt.
    - Moral: Keeping an open-source project work needs to spend time to maintain and develop new functions. We should find some ways to sustain instead of your enthusiasm only.

- Three support ways
    - **Donation**: (The directly method, haha.)
    - **Star**
    - **share**

- So, how to help someone who wants to support your project conveniently? 
  - **[ECPay](https://github.com/TLYu0419/facebook_crawler) website provides many ways to let people donate quickly**, such as credit cards, ATMs, convenience stores. And it accepts domestics and international donation. 
  - And **it’s very swift.** When you assign a new account, it will support ATM and convenience store donation methods. And two days later, it will permit domestic credit cards. And then, more two days later, international credit cards will be accepted!
      - But, **it will deduct some handling fees from the banks and the website.** For example, if someone donates NTD 1,000, the bank will take about 3% fees(bank) and NTD 1 (website).

## Package your project
- License	
  - An open-source project needs copyright. If not, nobody can copy, distribute, or modify your project in law.
  - How to choose a license?
    - **Everything is simple until the project can be profitable...**
    - There are many options we can choose from the Github website, such as GPL, BSD, Apache 2.0, MIT...
    - We can see the difference on license through the following picture. The most well-known are MIT and APACHE licenses. 
    - [一張圖看懂開源許可協議，開源許可證GPL、BSD、MIT、Mozilla、Apache和LGPL的區別【轉】](https://www.itread01.com/content/1545041946.html)
       ![](https://img-blog.csdn.net/20140811173721234?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGVzdGNzX2Ru/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

- Package steps
    1. Create a project on the Github website, and choose a license.
    2. Write a readme.md file to explain how to utilize your project.
       - It would be great if there are an introduction, contact information, and a quick-start in it.
    4. Write a setup.py file.
       - Describe the python environment you used and the dependency packages. 
    5. Create an account on the PyPi website.
       - https://test.pypi.org/
       - https://pypi.org/
    6. After you are all setting, you can package your project as follow methods:
       ```python
       python3 -m pip install --upgrade build
       python3 -m pip install --upgrade twine
       python3 -m build
       python3 -m twine upload --repository testpypi dist/*
       # python3 -m twine upload --repository pypi dist/*
       ```

   - More information:
     - [Packaging Python Projects](
     https://packaging.python.org/tutorials/packaging-projects/)
     - [如何开发自己的 Python 库](https://zhuanlan.zhihu.com/p/60836179)
     

- 
- [Using GitHub Actions to publish releases to pip/pypi (Developer demo) - YouTube](https://www.youtube.com/watch?v=U-aIPTS580s)