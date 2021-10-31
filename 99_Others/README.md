# 其他

## ubuntu

1. 安裝WSL2
   - [Ubuntu 20.04 LTS 桌面版详细安装指南](https://www.sysgeek.cn/install-ubuntu-20-04-lts-desktop/)
   - [Window10开发环境搭建(1) | 详细演示WSL2的安装](https://www.youtube.com/watch?v=BEVcW4kz1Kg)
     - [Install WSL | Microsoft Docs](https://docs.microsoft.com/en-us/windows/wsl/install)
2. 設定IDE
   - [Windows10开发环境搭建(2) | Terminal和VS Code](https://www.youtube.com/watch?v=0NjYngJ0HB0)

3. 客製化設定

   ```linux
   sudo apt update
   
   sudo apt install zsh
   
   sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
   
   git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
   
   # Find .zshrc file in ~/.zshrc folder
   # And change the ZSH_THEME to "powerlevel10k/powerlevel10k" 
   # ZSH_THEME="powerlevel10k/powerlevel10k"
   
   ```

   

   - [Windows10开发环境搭建(4) | 给WSL配置漂亮且强大的zsh - YouTube](https://www.youtube.com/watch?v=1fFWHyzYWls)

     - [zsh-users/zsh-autosuggestions: Fish-like autosuggestions for zsh (github.com)](https://github.com/zsh-users/zsh-autosuggestions)
     - [ohmyzsh/ohmyzsh](https://github.com/ohmyzsh/ohmyzsh)
     - [Make your WSL or WSL2 terminal awesome - with Windows Terminal, zsh, oh-my-zsh and Powerlevel10k - YouTube](https://www.youtube.com/watch?v=235G6X5EAvM)
     - [romkatv/powerlevel10k: A Zsh theme (github.com)](https://github.com/romkatv/powerlevel10k)

   - QA

     - zsh: command not found: node?

       > -- add following script to .zshrc file
       >
       > export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
       > [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm

     - zsh: command not found: hexo?

       > -- add hexo/node_modules to path
       >
       > export PATH=$HOME/bin:/usr/local/bin:/home/tlyu0419/github/hexo/node_modules/.bin:$PATH

4. [Windows10开发环境搭建(5) | WSL+VS Code 搭建Python开发环境](https://www.youtube.com/watch?v=BX7XwxQ1xlQ)

- Ref
  - [Ubuntu 20.04 LTS](https://www.microsoft.com/zh-tw/p/ubuntu-2004-lts/9n6svws3rx71)
  - [如何從Win10訪問Linux子系統中的文件](https://kknews.cc/tech/5v46vv2.html)
  - [[ Tools ] 在 Ubuntu 安裝 Node.js](https://oranwind.org/post-post-11/)
  - Ref
    - 
    - - 
    - 

## linux

- 雖然用 ubuntu 的方式可以直接在 win10 下建立 linux 的子系統，但是在記憶體不多(像我的電腦只有8G記憶體)的情況下，win10本身就直接吃掉了30%。除了會影響你的效能外，更重要的是這影響了我們美麗的心情。
- 因此我們也可以考慮直接把電腦的作業係統改成linux~
- - 



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
twine upload --skip-existing dist/*
```



