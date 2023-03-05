# Setting your own Python workstation with Visual Studio Code

Return to the [castle](https://github.com/Nkluge-correa/teeny-tiny_castle).

`Python` is a high-level, interpreted programming language that is widely used in various fields such as `data science`, `web development`, `artificial intelligence`, and `automation`. It was first released in 1991 and has since become one of the most popular programming languages due to its simple syntax, ease of use, and vast libraries of pre-built modules that make development faster and more efficient. Overall, `Python` is a versatile and powerful programming language that continues to gain popularity among developers of all levels.

![python](https://i.gifer.com/origin/25/25dcee6c68fdb7ac42019def1083b2ef_w200.gif)

## Checking if you have python

First of all, you can check if you already have Python installed in your machine by running the following line on your CMD (Microsoft Windows command):

```bash

python --version

```

You can access your CMD by writting _'cmd'_ on your windows start menu (_or just press the 'command' or 'windows logo' key on your keyboard_).

If you have python installed, you should get a response like this:

```bash

C:\Users\Yourname>python --version
Python 3.9.13

```

If you dont have python installed, you will receive a message like this:

```bash

C:\Users\Yourname>python --version
'python' is not recognized as an internal or external command,
operable program or batch file.

```

Also, you might get this message if _python's directory has not been added to the path of the Environment Variables_. In this case, the easy solution is just to uninstall python, and install it again the right way.

## Installing Python on Windows 11

**Note:** _Instaling python on a MAC in not much different, but we are going to focus on windows in this tutorial, since [windows is the most used OS out there](https://en.wikipedia.org/wiki/Usage*share*of*operating*systems#:~:text=For%20desktop%20and%20laptop%20computers,US%20up%20to%203.2%25)._

First, go to the [official website of Python](https://www.python.org/).

<img src="https://www.saashub.com/images/app/screenshots/5/8b2bb80f56f4/landing-medium.jpg?1634523102" width=400 />

In this webstite, you can download the lattest python version out there, **but we dont recommend that**. Depending on when you read this, the latest python version may still be in beta (e.g., python 3.10 or python 3.11). So you should use a more **stable** version. We are currently living in the **stable age of Python 3.8 and python 3.9**, so we recommend using one these two versions.

In this tutorial, we are going to go with **python 3.9.13**

In the [python.org](https://www.python.org/) homepage, click on the **'Download'** button. Rigth bellow you are gonna see an option that says **'All releases'**. Click on it.

Now you will see the options for downloading active python releases:

<img src="https://res.cloudinary.com/practicaldev/image/fetch/s--yaogPnvc--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://dev-to-uploads.s3.amazonaws.com/uploads/articles/683r055znnot4t8zoyu0.png" width=400 />

Bellow the _'Looking for a specific release?'_ heading, scroll till you find the 3.9.13 release (_you can choose another if you want to. But this one works fine and its very stable_).

Click on the download button, and you are going to be redirected to the download page for this python version.

**Scroll til the end of the page** and select the propper installer for your machine. There are installers for _MAC_ and _Windows_. To choose the rigth installer, you have to know if your machine has a **32-bit** or **64-bit** processor.

To do this, follow these instructions

- Click on the **Start button** then choose **Settings**.
- Click on **System**.
- Scroll until you find the **'About'** option.
- Your System Type will be there.

<img src="https://cdn.nerdschalk.com/wp-content/uploads/2021/08/win-11-32bit-64bit-architecture-2.png" width=400 />

Chances are that you probably have a 64-bit system. But if you dont, remmember that **you can run 32-bit applications on 64-bit OS, but not the other way around.**. If you cant find a python version for your 32-bit machine, we recommend using a browser development environment, like [Google Colab](https://colab.research.google.com/?utm*source=scs-index) or [Jupyter](https://jupyter.org/).

Now, download your installer. We recommend downloading the windows installer. In this example case its the _'Windows x86-64 executable installer'_.

After you download the executable (a small ~ 30 MB file), run the excutable. For simmplicity and n00b users, we recommend the **'Install Now'** option, that already comes with alot of python tools, like a simple IDE (_Integrated development environment_), pip (_python package installer_), among other tools.

<img src="https://docs.python.org/pt-br/3.8/_images/win_installer.png" width=400 />

We also recommend that you choose the **'Add Python 3.9 to PATH'** option before you install it.

Adding Python to PATH makes it possible for you to **run (use) Python from your command prompt (cmd)**. This lets you access the Python shell from your command prompt. In simpler terms, you can run your code from the Python shell by just typing _"python"_ in the command prompt:

```bash

C:\Users\Yourname>place where i keep my python files>python hello*world.py
Hello World!

```

**Now, just proceed with the installation and you are done! After, if you run the `python --version` on your cmd, you will see that you have python installed on your machine!**

Python comes with a simple IDE based on the tkinter GUI toolkit (a python library/module). On the rest of this tutorial, we will show you how to install and setup a more robust and versitile IDE: **Visual Studio Code (VS Code).**

## Installing VS Code

**Visual Studio Code**, also commonly referred to as **VS Code**, is a source-code editor Microsoft for Windows Linux and macOS. It has a bunch of functionalities that make it, according to [Stack Overflow](https://en.wikipedia.org/wiki/Stack*Overflow "Stack Overflow") 2021 Developer Survey, **the most popular developer environment tool.**

To download VS Code, go to [code.visualstudio.com](https://code.visualstudio.com/) and click the download button.

<img src="https://www.pylenin.com/content/images/2021/08/vscode-page.png" width=400 />

From then on, it is a tipical windows installation. Accept the terms, go forward, and you are done. After the installation is fineshed, you will get the starting screen of VS Code.

![vs-code-start](https://code.visualstudio.com/assets/docs/getstarted/tips-and-tricks/getstarted_page.png)

On the left side of the window, there are a coulpe of icons. Click on the **"Extensions"** icon (_a square with its top right quadrant removed_). You can also use the shortcut `control + shift + x`.

Look for the _'Python'_ extension, which will allow you to run python scripts on VS Code.

<img src="https://code.visualstudio.com/assets/docs/editor/extension-marketplace/extensions-python.png" width=400 />

**Feel free to explore the extensions available, [here](https://x-team.com/blog/best-vscode-extensions/) you can find some preety neat ones.** In this tutorial, we'll just show you how to get some of the big (usefull) ones.

To install extensions, you just need to click the button **'Install'** on the bottom-right corner of the intended extension.

We also recommend you to get the **Jupyter** extension, wich will allow you to run python code on a small "custom-jupyter-browser-window" inside VS Code.

<img src="https://user-images.githubusercontent.com/11532015/57962872-2adcc700-794f-11e9-8d7c-8019b6e962b5.png" width=400 />

Now that you have the **Python** and **Jupyter** extensions installed. Let's make a test-drive.

- Open VS Code.
- Create a folder (all name it `python project`).
- Create a file (name the file `hello.py`).
- Remember to use the `.py` extension, since it is what tells the computer _this is a python file_, just like `document.docx` is the ay of telling the computer that _this is a word file_.
- writte `print('hello world!')` on this file.
- Save the file (`control + s`).

There are three ways to run this file:

- Press the `Play` button on the top right corner of your VS Code station. A terminal is gonna pop up bellow your main screen, with the following:

```bash

PS C:\Users\Yourname\python project> & C:/Users/Yourname/AppData/Local/Programs/Python/Python38/python.exe "C:/Users/Yourname/python project/hello.py"
hello world!
PS C:\Users\Yourname\python project>

```

- Select your own line code with your mouse, `print('hello world!')`, and press `shift + enter`. This will run the selected code in your VS Code Jupyter terminal. To allow this feature, go to the tab `Files` (top-left corner), `Preferences > Settings`, writte `Interactive` on the search bar, and mark the box that allows for interactive window. Also, search for `Jupyter` in the seach bar, and see if all configurations that allow for 'interactive window' are marked (_usually it is_).

<img src="https://user-images.githubusercontent.com/24854248/113336891-b12a0880-932f-11eb-8774-05d8f14010e7.gif" width=400 />

- Run your saved python file, `hello.py`, on the VC Code terminal, using the command `python hello.py`.

```bash

PS C:\Users\Yourname\python project> python hello.py
hello world!
PS C:\Users\Yourname\python project>

```

You can open a terminal by clicking the `Terminal` tab on the top of your VC Code station.

**Now you are ready to start working with python on your own machine!**

## Final touches

- You can change some preferences (like _font-size, auto-save, mini-map_) by opening the tab `Files` (top-left corner), `Preferences > Settings`.
- You can run speciic lines, variables, blocks of code, and even a whole python script in the `interactive window`, just by selecting what you want to run an pressing `shift + enter`.
- There are various shortcut-keys to opoimize your time and work. You can see all of them by pressing `control + shift + p`.
- You can change the theme of your VS Code station by opening the tab `Files` (top-left corner), `Preferences > Color Themes`. Choose the one that fits your style. _Note: dark themes are better for your eyes._

---

Return to the [castle](https://github.com/Nkluge-correa/teeny-tiny_castle).
