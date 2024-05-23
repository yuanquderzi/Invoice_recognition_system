# -*- coding: utf-8 -*-

import time, os, datetime
from loguru import logger as log
from depoly.routers import router
from fastapi import FastAPI, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException
import uvicorn, traceback
from common.exceptions import ParsingError

app=FastAPI(
    title="发票识别",
    version="v1.0",
)
app.include_router(router,prefix="")

@app.middleware('http')
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    timestamp = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    response.headers['taskID'] = str(request.path_params.get('taskID'))
    response.headers['timestamp'] = timestamp
    response.headers['timecost'] = f"{(process_time * 1000):0.0f}ms"
    return response


# API接口，传入参数检验失败
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    log.error(traceback.format_exc())
    # print(str(exc.body))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"isSuccess":False,"code":1,"message":"Missing or wrong parameter.","result":""}
    )

# 图片解析失败
@app.exception_handler(ParsingError)
async def image_parser_error_exception_handler(request: Request, exc: ParsingError):
    log.error(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_451_UNAVAILABLE_FOR_LEGAL_REASONS,
        content={"isSuccess":False,"code":exc.code,"message":exc.message,"res":""}
    )

# 404 Not Found
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    log.error(traceback.format_exc())
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND,
      content={"isSuccess":False,'code':3,'message':'Not Found.','res':''})

# 预料之外的错误
@app.exception_handler(Exception)
async def unexcept_exception_handler(request: Request, exc: Exception):
    log.error(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"isSuccess":False,"code":4,"message":"Internal Error.","res":""}
    )

if __name__ == '__main__':
    uvicorn.run(app='api:app', host="0.0.0.0", port=8018, reload=True) # , workers = None, reload=True, debug=True)

