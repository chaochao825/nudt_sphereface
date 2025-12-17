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
    SSE print with standardized format matching face_json specification
    
    Args:
        event: Event name
        data: Legacy data dict (for backward compatibility)
        progress: Progress percentage (1-100)
        message: Human-readable message (preferably in Chinese)
        log: Log message with progress indicator
        callback_params: Callback parameters dict (task_run_id, method_type, etc.)
        details: Additional details dict
    """
    # Build standard SSE response format
    response = {
        "resp_code": 0,
        "resp_msg": "操作成功",
        "time_stamp": datetime.now().strftime("%Y/%m/%d-%H:%M:%S:%f")[:-3],
        "data": {
            "event": event
        }
    }
    
    # Add callback_params if provided (important for face_json format)
    if callback_params:
        response["data"]["callback_params"] = callback_params
    
    # Add progress if provided
    if progress is not None:
        response["data"]["progress"] = progress
    
    # Add message if provided
    if message:
        response["data"]["message"] = message
    
    # Add log if provided
    if log:
        response["data"]["log"] = log
    elif progress is not None and message:
        # Auto-generate log from progress and message
        response["data"]["log"] = f"[{progress}%] {message}\n"
    
    # Add details
    if details:
        response["data"]["details"] = details
    elif data and event not in ["input_path_validated", "output_path_validated", 
                                  "input_data_validated", "input_model_validated"]:
        # For backward compatibility, use data as details
        response["data"]["details"] = data
    else:
        # For validation events, merge data into response data
        response["data"].update(data)
    
    json_str = json.dumps(response, ensure_ascii=False, default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj)
    message_str = f"data: {json_str}\n\n"
    print(message_str, flush=True)
    return message_str

def sse_input_path_validated(args):
    try:
        if os.path.exists(args.input_path):
            sse_print("input_path_validated", {
                "status": "success",
                "message": "输入路径验证成功",
                "file_name": args.input_path
            }, progress=5, message="输入路径验证成功")
            
            try:
                if os.path.exists(f'{args.input_path}/data'):
                    data_files = glob.glob(os.path.join(f'{args.input_path}/data', '*/'))
                    sse_print("input_data_validated", {
                        "status": "success",
                        "message": "输入数据文件验证成功",
                        "file_name": data_files[0] if data_files else f'{args.input_path}/data'
                    }, progress=10, message="输入数据验证成功")
                else:
                    raise ValueError('输入数据文件未找到')
            except Exception as e:
                sse_print("input_data_validated", {"status": "failure", "message": f"{e}"})
                
            try:
                if os.path.exists(f'{args.input_path}/model'):
                    model_files = glob.glob(os.path.join(f'{args.input_path}/model', '*'))
                    sse_print("input_model_validated", {
                        "status": "success",
                        "message": "输入模型文件验证成功",
                        "file_name": model_files[0] if model_files else f'{args.input_path}/model'
                    }, progress=15, message="输入模型验证成功")
                else:
                    raise ValueError('输入模型文件未找到')
            except Exception as e:
                sse_print("input_model_validated", {"status": "failure", "message": f"{e}"})
        else:
            raise ValueError('输入路径未找到')
    except Exception as e:
        sse_print("input_path_validated", {"status": "failure", "message": f"{e}"})

def sse_output_path_validated(args):
    try:
        if os.path.exists(args.output_path):
            sse_print("output_path_validated", {
                "status": "success",
                "message": "输出路径验证成功",
                "file_name": args.output_path
            }, progress=20, message="输出路径验证成功")
        else:
            raise ValueError('输出路径未找到')
    except Exception as e:
        sse_print("output_path_validated", {"status": "failure", "message": f"{e}"})

def sse_adv_samples_gen_validated(adv_image_name, current, total):
    progress_pct = int(25 + (current / total) * 60)  # 25-85% range for sample generation
    sse_print("adv_samples_gen_validated", {
        "status": "success",
        "message": f"对抗样本已生成: {os.path.basename(adv_image_name)}",
        "file_name": adv_image_name,
        "current": current,
        "total": total
    }, progress=progress_pct, message=f"生成对抗样本 ({current}/{total})")

def sse_clean_samples_gen_validated(clean_image_name, current, total):
    progress_pct = int(25 + (current / total) * 60)  # 25-85% range for sample generation
    sse_print("clean_samples_gen_validated", {
        "status": "success",
        "message": f"防御样本已生成: {os.path.basename(clean_image_name)}",
        "file_name": clean_image_name,
        "current": current,
        "total": total
    }, progress=progress_pct, message=f"处理防御样本 ({current}/{total})")

def sse_epoch_progress(progress, total, epoch_type="Epoch"):
    progress_pct = int((progress / total) * 100)
    sse_print("training_progress", {
        "progress": progress,
        "total": total,
        "type": epoch_type
    }, progress=progress_pct, message=f"{epoch_type} {progress}/{total}")

def sse_error(message, event_name="error"):
    sse_print(event_name, {"status": "failure", "message": message})

def save_json_results(results: dict, output_path: str, filename: str = "results.json"):
    """Save results to JSON file"""
    os.makedirs(output_path, exist_ok=True)
    json_path = os.path.join(output_path, filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=lambda obj: obj.item() if isinstance(obj, np.generic) else float(obj) if isinstance(obj, (np.floating, np.integer)) else obj)
    return json_path
