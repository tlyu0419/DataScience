# 其他

## ubuntu

- 很多語法基於 linux 來開發的語法在 windows 上會遇到不能執行的指令，雖然可以安裝很多的 windows 開發者工具來盡可能處理，但還是有解不完的bug
- 這時候就可以直接在windows 10 上安裝 linux 的子系統 -- ubuntu，直接在windows下建立linux的環境來執行程式~
- Ref
  - [Ubuntu 20.04 LTS](https://www.microsoft.com/zh-tw/p/ubuntu-2004-lts/9n6svws3rx71)
  - [如何從Win10訪問Linux子系統中的文件](https://kknews.cc/tech/5v46vv2.html)
  - [How to Install Miniconda on Ubuntu 20.04](https://varhowto.com/install-miniconda-ubuntu-20-04/)
  - [[ Tools ] 在 Ubuntu 安裝 Node.js](https://oranwind.org/post-post-11/)

## linux

- 雖然用 ubuntu 的方式可以直接在 win10 下建立 linux 的子系統，但是在記憶體不多(像我的電腦只有8G記憶體)的情況下，win10本身就直接吃掉了30%。除了會影響你的效能外，更重要的是這影響了我們美麗的心情。
- 因此我們也可以考慮直接把電腦的作業係統改成linux~
- Ref
  - [Ubuntu 20.04 LTS 桌面版详细安装指南](https://www.sysgeek.cn/install-ubuntu-20-04-lts-desktop/)
  - [Window10开发环境搭建(1) | 详细演示WSL2的安装](https://www.youtube.com/watch?v=BEVcW4kz1Kg)
  - [Windows10开发环境搭建(2) | Terminal和VS Code](https://www.youtube.com/watch?v=0NjYngJ0HB0)
  - [Windows10开发环境搭建(5) | WSL+VS Code 搭建Python开发环境](https://www.youtube.com/watch?v=BX7XwxQ1xlQ)



### 打包專案

- [Packaging Python Projects](https://packaging.python.org/tutorials/packaging-projects/)

- [如何开发自己的 Python 库](https://zhuanlan.zhihu.com/p/60836179)

- [How to upload new versions of project to PyPI with twine?](https://stackoverflow.com/questions/52016336/how-to-upload-new-versions-of-project-to-pypi-with-twine)

- freeze package

  ```python
  pip freeze
  ```

  

```bash
# python3 -m build
python3 setup.py sdist bdist_wheel

# python3 -m twine upload --repository testpypi dist/*
#twine upload --skip-existing dist/*
```



