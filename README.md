# fund-research-agent

基金产品识别、底层基金归并、官方资料/月报搜索、社媒痛点与卖点分析工具。

## 功能

1. 读取图片、Excel、CSV，抽取基金产品代码、晨星代码、ISIN、中英文基金名称。
2. 归并同一底层基金的不同份额，统计底层基金数量、份额数量、份额代码和名称。
3. 搜索每只底层基金的最新 factsheet、月报、KFS、产品资料概要等官方资料。
4. 搜索小红书、微博、抖音等公开网页结果，或读取用户导出的评论 CSV，分析投资者痛点和卖点。
5. 输出 Excel 研究报告。

> 说明：小红书、微博、抖音的评论区通常需要登录且有反爬限制。本项目不绕过登录、验证码或平台访问限制；推荐使用平台授权接口、合规第三方数据服务，或用户自行导出的评论 CSV。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

图片 OCR 依赖较重；如只读取 Excel/CSV，可以先不安装 `paddleocr` / `paddlepaddle`。

## 配置

复制环境变量模板：

```bash
cp .env.example .env
```

填写搜索和大模型相关 API Key：

```bash
SERPAPI_KEY=your_key
OPENAI_API_KEY=your_key
```

没有 API Key 也可以运行本地表格读取、基金识别、底层归并和统计。

## 使用

读取 Excel：

```bash
python main.py data/fund_list.xlsx
```

读取多张图片：

```bash
python main.py image1.jpeg image2.jpeg image3.jpeg image4.jpeg
```

只做本地识别和归并，不做联网搜索：

```bash
python main.py data/fund_list.xlsx --no-doc-search --no-social-search
```

接入已导出的社媒评论 CSV：

```bash
python main.py data/fund_list.xlsx --social-comments-csv data/social_comments.csv
```

输出文件默认在：

```text
output/fund_research_report.xlsx
```

## 社媒评论 CSV 格式

```csv
product_code,platform,user_text,publish_time,like_count,url
968000.OF,小红书,这个基金分红挺高但净值也跌了，不知道是不是亏本金,2025-10-12,23,https://...
968001.OF,微博,亚洲债券最近是不是受降息预期影响比较大,2025-11-03,8,https://...
```

## 报告 Sheet

| Sheet | 内容 |
|---|---|
| 全部份额 | 每个基金份额代码、晨星代码、ISIN、中英文名、归属底层基金 |
| 底层基金统计 | 每只底层基金、份额数量、对应产品代码、ISIN、名称集合 |
| 官方资料搜索结果 | 每只底层基金搜索到的 factsheet / 月报 / KFS 链接 |
| 官方资料分析 | 产品分类、特色、罕见策略、同类对比角度 |
| 社媒搜索结果 | 小红书 / 微博 / 抖音公开搜索结果 |
| 社媒痛点卖点 | 每个份额代码的投资者痛点、卖点、高频信号 |
