# fund-research-agent

基金产品识别、底层基金归并、官方资料/月报搜索、社媒痛点与卖点分析工具。

## 功能

1. 读取图片、Excel、CSV、PDF、TXT/Markdown 附件，抽取基金产品代码、晨星代码、ISIN、中英文基金名称。
2. 归并同一底层基金的不同份额，统计底层基金数量、份额数量、份额代码和名称。
3. 深度搜索每只底层基金的最新 factsheet、月报、KFS、产品资料概要等资料，抓取正文摘录、识别资料日期、按官方来源/资料类型/新鲜度打分，标记最新候选资料。
4. 在基金列表内按同类产品做对比，分析每只基金的产品特色、差异化或罕见投资策略，并保留资料链接证据。
5. 搜索小红书、微博、抖音近半年公开网页结果，或读取用户导出的评论 CSV，过滤近半年评论，分析投资者痛点和内容卖点。
6. 输出 Excel 研究报告。

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
TAVILY_API_KEY=your_key
OPENAI_API_KEY=your_key
```

没有搜索 API Key 也可以运行本地附件读取、基金识别、底层归并、统计和用户导出评论分析。设置 `FETCH_DOCUMENT_TEXT=0` 可只保留搜索结果，不抓取资料正文。

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

读取 PDF/TXT 附件：

```bash
python main.py attachment.pdf notes.txt --no-social-search
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

输出文件默认在：

```text
output/fund_research_report.xlsx
```

## 社媒评论 CSV 格式

```csv
product_code,platform,user_text,publish_time,like_count,url
968000.OF,小红书,这个基金分红挺高但净值也跌了，不知道是不是亏本金,2026-03-12,23,https://...
968001.OF,微博,亚洲债券最近是不是受降息预期影响比较大,2026-04-03,8,https://...
```

`publish_time` 会用于近半年过滤；没有日期的公开搜索结果会保留，但报告会标记日期来源为 `query_window_only`，提醒需要复核。

## 报告 Sheet

| Sheet | 内容 |
|---|---|
| 基金清单摘要 | 附件中识别出的基金产品数量、底层基金数量、基金名称、产品代码和 ISIN 汇总 |
| 全部份额 | 每个基金份额代码、晨星代码、ISIN、中英文名、归属底层基金 |
| 底层基金统计 | 每只底层基金、份额数量、对应产品代码、ISIN、名称集合 |
| 官方资料搜索结果 | 每只底层基金搜索到的 factsheet / 月报 / KFS 链接、来源质量、资料日期、最新候选、正文摘录 |
| 官方资料分析 | 产品分类、最新资料证据、产品特色、罕见策略、列表内同类对比 |
| 社媒搜索结果 | 小红书 / 微博 / 抖音公开搜索结果或用户导出评论，含近半年标记 |
| 社媒痛点卖点 | 每个份额代码的投资者痛点、卖点、高频信号、样本片段 |

## 实现边界

- 官方资料搜索使用 SerpAPI 和 Tavily；同时配置时会合并去重。系统会优先识别官方/基金经理域名，但最终资料有效性仍应由研究员复核。
- 最新资料通过标题、链接、摘要和抓取正文中的日期识别，并结合资料类型和来源质量评分；无法识别日期的资料不会被强行当作最新。
- 社媒搜索只使用公开网页结果和用户提供的导出评论，不绕过平台登录、验证码或反爬机制。若需要真实评论区全量样本，应接入平台授权接口或合规第三方数据服务。
