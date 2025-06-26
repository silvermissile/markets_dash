##  升级 Python
``` sh 
conda create -n markets_dash python=3.12
pip install -r requirements.txt
python my_app.py 
```

## 使用本地 CSS
下载 CSS  文件到本地
https://bootswatch.com/lux/


Dash 框架会自动加载 `assets/` 目录下的所有 CSS 文件。你需要：

1. 在项目根目录下新建一个名为 `assets` 的文件夹。
2. 把你需要的 CSS 文件（如 `bootstrap_lux.min.css`、`dbc.min.css` 等）放到 `assets/` 文件夹里。

**正确结构示例：**
```
project_markets_dash/
│
├── assets/
│   ├── bootstrap_lux.min.css
│   └── dbc.min.css
├── my_app.py
├── ...
```
