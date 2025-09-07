# PDF-Excel 转换服务（各类交易流水）

本项目提供将“各类交易流水”PDF 转换为 Excel 的服务，支持网页上传与下载，也支持命令行/接口调用。已在微信/支付宝基础上扩展为通用流水识别（银行对账单/平台流水等），优化多页表头继承、字段映射与去重，优先使用轻量解析以提升速度与稳定性。

## 功能概览
- Web 服务：上传 PDF，返回 XLSX 下载（内置简易前端页面）。
- 支持对象：
  - 微信支付、支付宝交易明细（常见版式）。
  - 各类银行/平台流水（含“记账日期/交易日期/借方/贷方/余额/摘要/对方账户”等版式）。
- 多页识别：若后续页不再重复表头，自动继承上一页表头，避免只识别第一页。
- 字段标准化：统一输出字段命名，并在存在同名/重复列时自动合并。
- 轻量优先：解析顺序 pdfplumber → tabula → camelot，减少不必要的 Java/OCR 开销。
- 汇总页：若可推断收/支为带符号金额，生成 Summary 工作表。

## 目录与关键文件
- 服务入口（后端）：`app/server.py`
- 前端页面：`web/index.html`
- 核心转换逻辑：`wechat_pdf2excel_ocr.py`
- 容器构建：`Dockerfile`
- 本地编排：`docker-compose.yml`

## 快速开始
1) 构建镜像（推荐镜像名）
- `docker build -t pdf-excel:web .`

2) 运行 Web 服务（宿主 8003 → 容器 8003）
- `docker run -d --name pdf-excel -p 8003:8003 -e JAVA_TOOL_OPTIONS='-Xms256m -Xmx2g' pdf-excel:web`
- 打开浏览器：`http://localhost:8003`
- 健康检查：`curl http://localhost:8003/healthz`

3) 上传转换
- 网页上可选择一个或多个 PDF：
  - 单个文件：直接下载 XLSX
  - 多个文件：逐个请求 /convert 并分别下载多个 XLSX（不再打包 ZIP）

4) 直接调用 API
- 单个：
  - `curl -F "file=@input/微信支付交易明细.pdf" http://localhost:8003/convert -o output/wechat.xlsx`
- 批量（可选 ZIP 打包）：
  - `curl -F "files=@input/微信支付交易明细.pdf" -F "files=@input/支付宝交易明细(20240930-20250830).pdf" http://localhost:8003/convert-batch -o output/converted.zip`
- 健康检查：
  - `curl http://localhost:8003/healthz`

5) 一次性 CLI（无需启动 Web）
- `docker run --rm --entrypoint python -v "$PWD/input:/data/in" -v "$PWD/output:/data/out" pdf-excel:web /app/wechat_pdf2excel_ocr.py "/data/in/文件.pdf" -o "/data/out/result.xlsx"`

6) docker-compose（可选）
- 启动：`docker compose up --build -d`
- 访问：`http://localhost:8003`

## API 说明
- `POST /convert`
  - 请求：`multipart/form-data`，字段名 `file`，值为单个 PDF 文件。
  - 返回：`application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`，浏览器会下载。
- `POST /convert-batch`
  - 请求：`multipart/form-data`，字段名可重复的 `files`，值为多个 PDF 文件。
  - 返回：`application/zip`，将所有转换后的 XLSX 打包为 ZIP；如有失败，ZIP 内含 `errors.txt`。
- `GET /healthz`
  - 返回：`{"status":"ok"}`

服务实现见 `app/server.py`，前端页面见 `web/index.html`。

## 解析与字段说明
核心逻辑位于 `wechat_pdf2excel_ocr.py`。
默认导出保留原表字段名（不做字段改名/英文化）。汇总页若可推断带符号金额，会额外生成，但不影响明细表的字段名。

- 解析顺序：
  1. `pdfplumber`（优先，轻量、稳定）
  2. 失败回退 `tabula`
  3. 再回退 `camelot`

- 表头识别：
  - 在每页表格的前 6 行内寻找表头（兼容“微信/支付宝/通用银行流水”表头）。
  - 若本页未找到表头但与上一页列数一致，则继承上一页表头。

- 重复列合并：
  - 对同名列自动按“前者优先、缺失用后者填充”的方式合并，随后去重。

注：内部仍会做“标准化识别”用于准确抽取和汇总，但导出的明细页保留原字段名。若你确需导出标准化字段名，可在 CLI 使用 `--std-headers` 开关。

- 带符号金额与汇总：
  - 若存在 `direction` 与 `amount`，则生成 `amount_signed`（收入为正、支出为负）。
  - 若存在“借方/贷方”或“收入/支出”分列，自动生成带符号金额并补齐 `direction/amount`。
  - 生成工作表：
    - `Transactions`：合并后的明细数据
    - `Summary`：按收入/支出汇总（若可计算 `amount_signed`）

- 去重策略：
  - 如同时存在 `transaction_id/timestamp/amount_signed`，将其拼接去重，减少重复记录。
  - 否则回退以 `(timestamp, amount, counterparty, note)` 近似去重。

## 性能与资源建议
- 资源限制：推荐运行时设置 `JAVA_TOOL_OPTIONS='-Xms256m -Xmx2g'`，避免 Tabula 过度占用内存。
- OrbStack/Mac：宿主可用内存建议 6–8GB；你已设为 8GB。
- 大文件：首次解析耗时更久。若需要，我可以在 API/CLI 增加 `pages`（分页处理）与 `workers`（并发）参数以进一步提速。

## 常见问题与排查
- 上传报错 `Form data requires "python-multipart"`：
  - 已在 `requirements.txt` 增加 `python-multipart`，若出现请重新构建镜像。
- 容器端口已占用：
  - 变更 `-p 8003:8000` 或停止占用进程；compose 中修改 `docker-compose.yml` 的映射端口。
- 只识别第一页：
  - 已通过“表头继承”修复；若遇到版式差异导致列数变化，请提供样页以便细化规则。
- 支付宝列缺失或冲突：
  - 已扩展常见列名变体并合并重复列；如仍缺，请给出样例列名（截图/文本）以便补充映射规则。
- OOM / 退出码 137：
  - 增加宿主内存（OrbStack）、限制 JVM 堆（见上）、优先轻量路径（已默认）。
- 查看日志：`docker logs -f pdf-excel`

## 维护与开发
- 当前分支：`feature/next`（已包含 Web 化改造与解析修复）。
- 主文件：
  - 后端：`app/server.py`
  - 前端：`web/index.html`
  - 解析：`wechat_pdf2excel_ocr.py`
  - 容器：`Dockerfile`
  - 编排：`docker-compose.yml`
- 重建镜像：`docker build -t pdf-excel:web .`
- 启动服务：`docker run -d --name pdf-excel -p 8003:8003 -e JAVA_TOOL_OPTIONS='-Xms256m -Xmx2g' pdf-excel:web`

## 后续可选优化（Roadmap）
- API/CLI 参数：`type=wechat|alipay|auto`、`engine=pdfplumber|tabula|camelot|auto`、`pages=1-100`、`workers=N`。
- 批量上传与打包下载；进度与排队可视化。
- 更细粒度字段：支付宝“对方账号/名称”分列、凭证号/渠道订单号拆分等。
- 解析模板配置化与单元测试覆盖。
