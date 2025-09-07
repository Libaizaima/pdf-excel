from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
import tempfile, os, pathlib, zipfile
from typing import List

# 复用已有转换逻辑
from wechat_pdf2excel_ocr import wechat_pdf_to_excel

app = FastAPI(title="PDF → Excel 转换服务", version="1.1.0")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


def _safe_filename(name: str, suffix: str) -> str:
    stem = pathlib.Path(name).stem or "result"
    return f"{stem}{suffix}"


@app.get("/")
def index() -> HTMLResponse:
    index_path = "/app/web/index.html"
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>Service is running</h1>", status_code=200)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read(), status_code=200)


@app.post("/convert")
async def convert_pdf(file: UploadFile = File(...), background: BackgroundTasks = None):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    # 将上传内容落地到临时文件
    in_tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        content = await file.read()
        in_tmp.write(content)
        in_tmp.flush()
    finally:
        in_tmp.close()

    # 输出临时路径
    out_name = _safe_filename(file.filename, ".xlsx")
    out_tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False).name

    try:
        # 调用原有转换函数
        wechat_pdf_to_excel(in_tmp.name, out_tmp)
    except Exception as e:
        # 清理并返回错误
        try:
            os.remove(in_tmp.name)
        except Exception:
            pass
        try:
            if os.path.exists(out_tmp):
                os.remove(out_tmp)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"转换失败: {e}")

    # 使用后台任务在响应发送后清理临时文件
    if background is not None:
        background.add_task(lambda p: os.path.exists(p) and os.remove(p), in_tmp.name)
        background.add_task(lambda p: os.path.exists(p) and os.remove(p), out_tmp)

    return FileResponse(
        out_tmp,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=out_name,
    )

@app.post("/convert-batch")
async def convert_batch(files: List[UploadFile] = File(...), background: BackgroundTasks = None):
    if not files:
        raise HTTPException(status_code=400, detail="请上传至少一个 PDF 文件")

    # 临时目录用于中间产物
    workdir = tempfile.TemporaryDirectory()
    workpath = workdir.name
    out_paths = []
    tmp_inputs = []
    errors = []

    # 处理每个文件
    for f in files:
        name = f.filename or "document.pdf"
        if not name.lower().endswith(".pdf"):
            errors.append(f"不支持的文件类型: {name}")
            continue
        in_path = os.path.join(workpath, _safe_filename(name, ".pdf"))
        with open(in_path, "wb") as w:
            w.write(await f.read())
        tmp_inputs.append(in_path)
        out_name = _safe_filename(name, ".xlsx")
        out_path = os.path.join(workpath, out_name)
        try:
            wechat_pdf_to_excel(in_path, out_path)
            out_paths.append(out_path)
        except Exception as e:
            errors.append(f"{name}: 转换失败 - {e}")

    if not out_paths and errors:
        # 全部失败
        workdir.cleanup()
        raise HTTPException(status_code=500, detail="; ".join(errors))

    # 打包为 ZIP
    zip_tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False).name
    with zipfile.ZipFile(zip_tmp, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out_paths:
            arcname = os.path.basename(p)
            zf.write(p, arcname)
        if errors:
            # 放入错误说明文本
            err_txt = os.path.join(workpath, "errors.txt")
            with open(err_txt, "w", encoding="utf-8") as ef:
                ef.write("\n".join(errors))
            zf.write(err_txt, "errors.txt")

    # 清理任务：临时目录和 zip
    if background is not None:
        background.add_task(lambda d: d.cleanup(), workdir)
        background.add_task(lambda p: os.path.exists(p) and os.remove(p), zip_tmp)

    return FileResponse(
        zip_tmp,
        media_type="application/zip",
        filename="converted_excels.zip",
    )
