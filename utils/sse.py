import json
import os
import sys
import glob
from pathlib import Path
import numpy as np
from datetime import datetime

def sse_print(event: str, data: dict, progress: int = None, message: str = None, log: str = None, 
              callback_params: dict = None, details: dict = None) -> str:
    """
    SSE print with strict format:
    event: event_name
    data: {"progress": ..., "message": ..., "log": ..., "callback_params": ..., "details": ...}
    """
    data_to_send = {}
    if callback_params: data_to_send["callback_params"] = callback_params
    if progress is not None: data_to_send["progress"] = progress
    if message: data_to_send["message"] = message
    if log: data_to_send["log"] = log
    elif progress is not None and message: data_to_send["log"] = f"[{progress}%] {message}\n"
    if details: data_to_send["details"] = details
    elif data and event not in ["input_path_validated", "output_path_validated", "input_data_validated"]:
        data_to_send["details"] = data
    else:
        if data: data_to_send.update(data)
    
    json_str = json.dumps(data_to_send, ensure_ascii=False, default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj)
    sys.stdout.write(f"event: {event}\n")
    sys.stdout.write(f"data: {json_str}\n\n")
    sys.stdout.flush()
    return f"event: {event}\ndata: {json_str}\n\n"

def sse_heartbeat(progress, message, callback_params=None):
    sse_print("progress_update", {}, progress=progress, message=message, callback_params=callback_params)

def sse_input_path_validated(args):
    try:
        if os.path.exists(args.input_path):
            sse_print("input_path_validated", {"status": "success", "message": "输入路径验证成功", "file_name": args.input_path}, progress=1, message="输入路径验证成功")
            if os.path.exists(f'{args.input_path}/data'):
                data_files = glob.glob(os.path.join(f'{args.input_path}/data', '*/'))
                sse_print("input_data_validated", {"status": "success", "message": "输入数据文件验证成功", "file_name": data_files[0] if data_files else f'{args.input_path}/data'}, progress=2, message="输入数据验证成功")
    except Exception: pass

def sse_output_path_validated(args):
    try:
        if os.path.exists(args.output_path):
            sse_print("output_path_validated", {"status": "success", "message": "输出路径验证成功", "file_name": args.output_path}, progress=3, message="输出路径验证成功")
    except Exception: pass

def sse_adv_samples_gen_validated(adv_image_name, current, total):
    progress_pct = int(25 + (current / total) * 60)
    sse_print("adv_samples_gen_validated", {"status": "success", "message": f"对抗样本已生成: {os.path.basename(adv_image_name)}", "file_name": adv_image_name, "current": current, "total": total}, progress=progress_pct, message=f"生成对抗样本 ({current}/{total})")

def sse_clean_samples_gen_validated(clean_image_name, current, total):
    progress_pct = int(25 + (current / total) * 60)
    sse_print("clean_samples_gen_validated", {"status": "success", "message": f"防御样本已生成: {os.path.basename(clean_image_name)}", "file_name": clean_image_name, "current": current, "total": total}, progress=progress_pct, message=f"处理防御样本 ({current}/{total})")

def sse_epoch_progress(progress, total, epoch_type="Epoch"):
    progress_pct = int((progress / total) * 100)
    sse_print("training_progress", {"progress": progress, "total": total, "type": epoch_type}, progress=progress_pct, message=f"{epoch_type} {progress}/{total}")

def sse_error(message, event_name="error"):
    sse_print(event_name, {"status": "failure", "message": message})

def save_json_results(results: dict, output_path: str, filename: str = "results.json"):
    os.makedirs(output_path, exist_ok=True)
    json_path = os.path.join(output_path, filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=lambda obj: obj.item() if isinstance(obj, np.generic) else float(obj) if isinstance(obj, (np.floating, np.integer)) else obj)
    return json_path

def sse_final_result(results: dict, event_name="final_result"):
    sse_print(event_name, {}, progress=100, message=results.get('message', '操作成功'), details=results)

def sse_class_number_validation(expected, got):
    sse_print("class_number_validated", {"status": "failure", "message": f"expect CLASS_NUMBER {expected} but got {got}"})
