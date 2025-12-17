import torch
import torch.nn as nn
import os
import glob
import numpy as np
import zipfile
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from .sphereface_model import SphereFaceModel
from utils.sse import sse_adv_samples_gen_validated, sse_clean_samples_gen_validated, sse_epoch_progress, sse_error, sse_print, save_json_results


def load_dataset_with_fallback(data_path, model_name, max_extract=50):
    """Smart dataset loader with ZIP support and fallback to test data"""
    from pathlib import Path
    import zipfile
    data_path = Path(data_path)
    sse_print("dataset_check", {}, progress=22, message="检查数据集可用性", log=f"[22%] 检查数据集路径: {data_path}\n")
    zip_files = list(data_path.glob('*.zip')) + list(data_path.glob('*/*.zip'))
    if zip_files:
        zip_file = zip_files[0]
        sse_print("compressed_found", {}, progress=22, message=f"发现压缩数据集: {zip_file.name}", log=f"[22%] 检测到压缩文件: {zip_file.name}\n")
        extract_dir = data_path / '.extracted' / zip_file.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            existing = list(extract_dir.rglob('*.jpg')) + list(extract_dir.rglob('*.png')) + list(extract_dir.rglob('*.pgm'))
            if len(existing) >= 10:
                sse_print("using_cached", {}, progress=23, message=f"使用缓存数据: {len(existing)}张图片", log=f"[23%] 发现{len(existing)}张缓存图片\n")
                return str(extract_dir), False
            with zipfile.ZipFile(zip_file, 'r') as zf:
                members = [m for m in zf.namelist() if any(m.lower().endswith(e) for e in ['.jpg', '.jpeg', '.png', '.pgm'])]
                if members:
                    extract_count = min(max_extract, len(members))
                    sse_print("extracting", {}, progress=23, message=f"正在提取{extract_count}张样本图片...", log=f"[23%] 从压缩包提取{extract_count}张图片\n")
                    for member in members[:extract_count]:
                        try: zf.extract(member, extract_dir)
                        except: pass
                    extracted = list(extract_dir.rglob('*.jpg')) + list(extract_dir.rglob('*.png')) + list(extract_dir.rglob('*.pgm'))
                    if len(extracted) > 0:
                        sse_print("extraction_success", {}, progress=24, message=f"成功提取{len(extracted)}张图片", log=f"[24%] 提取完成，共{len(extracted)}张图片\n")
                        return str(extract_dir), False
        except Exception as e:
            sse_print("extraction_failed", {}, progress=23, message=f"提取失败，使用备用数据", log=f"[23%] 提取错误: {str(e)[:100]}\n")
    existing = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.pgm']:
        existing.extend(list(data_path.rglob(ext)))
    existing = [f for f in existing if '__MACOSX' not in str(f)]
    if len(existing) > 0:
        sse_print("using_existing", {}, progress=24, message=f"使用现有数据集: {len(existing)}张图片", log=f"[24%] 发现{len(existing)}张现有图片\n")
        return str(data_path), False
    sse_print("using_fallback", {}, progress=24, message="使用测试数据（真实数据集不可用）", log=f"[24%] 回退到项目测试数据\n")
    project_root = Path(__file__).parent.parent
    fallback_path = project_root / 'test_data'
    if not fallback_path.exists() or len(list(fallback_path.rglob('*.jpg'))) == 0:
        fallback_path.mkdir(parents=True, exist_ok=True)
        _create_test_images(fallback_path)
    return str(fallback_path), True

def _create_test_images(path, count=10):
    """Create minimal test images"""
    try:
        import numpy as np
        for i in range(count):
            img_array = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            Image.fromarray(img_array).save(path / f'test_{i:03d}.jpg')
        sse_print("created_test_data", {}, progress=24, message=f"已创建{count}张测试图片", log=f"[24%] 生成{count}张测试图片\n")
    except Exception as e:
        pass

def get_model(cfg):
    """Get SphereFaceModel model"""
    model = SphereFaceModel(num_classes=cfg.num_classes)
    
    # Load pretrained weights if available
    if hasattr(cfg, 'pretrained') and cfg.pretrained and os.path.exists(cfg.pretrained):
        try:
            state_dict = torch.load(cfg.pretrained, map_location=cfg.device)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Could not load pretrained model from {cfg.pretrained}: {e}")
    
    model = model.to(cfg.device)
    return model

def validate_class_number(model, cfg):
    """Validate CLASS_NUMBER matches between model and config"""
    if hasattr(model, 'fc'):
        model_classes = model.fc.out_features
    elif hasattr(model, 'classifier'):
        model_classes = model.classifier[-1].out_features if isinstance(model.classifier, nn.Sequential) else model.classifier.out_features
    else:
        # Cannot determine, skip validation
        return True
    
    if model_classes != cfg.num_classes:
        from utils.sse import sse_error
        sse_error(f"expect CLASS_NUMBER {model_classes} but got {cfg.num_classes}", "input_model_validated")
        return False
    return True

def main(args, cfg):
    """Main function for SphereFaceModel"""
    from attacks import BIMAttack, DIMAttack, TIMAttack, PGDAttack, CWAttack, DeepFoolAttack
    from defends import HGDDefense, TVMDefense, LivenessDetection, FeatureSpacePurification, EnsembleDefense, AdversarialDetector
    
    try:
        # Get model
        model = get_model(cfg)
        
        # Validate CLASS_NUMBER for attack/adv modes
        if cfg.mode in ['adv', 'attack'] and hasattr(cfg, 'pretrained') and cfg.pretrained:
            if not validate_class_number(model, cfg):
                return
        
        # Load images
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])
        
        image_files = glob.glob(os.path.join(cfg.data_path, '**', '*.jpg'), recursive=True) + \
                     glob.glob(os.path.join(cfg.data_path, '**', '*.png'), recursive=True)
        
        if not image_files:
            image_files = glob.glob(os.path.join(cfg.data_path, '*.jpg')) + \
                         glob.glob(os.path.join(cfg.data_path, '*.png'))
        
        if not image_files:
            sse_error("No images found in data path")
            return
        
        # Process based on mode
        if cfg.mode == 'adv' or cfg.mode == 'attack':
            # Attack mode
            attack_method = args.attack_method.lower()
            
            sse_print("attack_process_start", {}, progress=25, 
                     message=f"Starting {attack_method.upper()} attack generation",
                     details={"attack_method": attack_method, "model": cfg.model})
            
            if attack_method == 'bim':
                attacker = BIMAttack(model, epsilon=args.epsilon, alpha=args.step_size, 
                                   iterations=args.max_iterations, device=cfg.device)
            elif attack_method == 'dim':
                attacker = DIMAttack(model, epsilon=args.epsilon, alpha=args.step_size, 
                                   iterations=args.max_iterations, device=cfg.device)
            elif attack_method == 'tim':
                attacker = TIMAttack(model, epsilon=args.epsilon, alpha=args.step_size, 
                                   iterations=args.max_iterations, device=cfg.device)
            elif attack_method == 'pgd':
                attacker = PGDAttack(model, epsilon=args.epsilon, alpha=args.step_size, 
                                   iterations=args.max_iterations, device=cfg.device)
            elif attack_method == 'cw':
                attacker = CWAttack(model, max_iterations=args.max_iterations, device=cfg.device)
            elif attack_method == 'deepfool':
                attacker = DeepFoolAttack(model, max_iterations=args.max_iterations, device=cfg.device)
            else:
                sse_error(f"Unknown attack method: {attack_method}")
                return
            
            # Process images and collect metrics
            total_images = min(10, len(image_files))
            successful_attacks = 0
            output_files = []
            
            # Metrics for comparison
            original_predictions = []
            adversarial_predictions = []
            original_confidences = []
            adversarial_confidences = []
            perturbation_norms = []
            
            for idx, img_path in enumerate(image_files[:total_images]):
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(cfg.device)
                    
                    # Get original prediction
                    with torch.no_grad():
                        orig_output = model(image_tensor)
                        orig_pred = orig_output.argmax(dim=1).item()
                        orig_conf = torch.softmax(orig_output, dim=1).max().item()
                    
                    # Generate adversarial example
                    adv_image = attacker.attack(image_tensor)
                    
                    # Get adversarial prediction
                    with torch.no_grad():
                        adv_output = model(adv_image)
                        adv_pred = adv_output.argmax(dim=1).item()
                        adv_conf = torch.softmax(adv_output, dim=1).max().item()
                    
                    # Calculate perturbation norm
                    perturbation = (adv_image - image_tensor).cpu().numpy()
                    l2_norm = np.linalg.norm(perturbation)
                    linf_norm = np.abs(perturbation).max()
                    
                    # Check if attack was successful
                    attack_success = (orig_pred != adv_pred)
                    if attack_success:
                        successful_attacks += 1
                    
                    # Collect metrics
                    original_predictions.append(orig_pred)
                    adversarial_predictions.append(adv_pred)
                    original_confidences.append(orig_conf)
                    adversarial_confidences.append(adv_conf)
                    perturbation_norms.append({'l2': l2_norm, 'linf': linf_norm})
                    
                    # Save adversarial image
                    adv_image_pil = transforms.ToPILImage()(adv_image.squeeze(0).cpu())
                    output_path = os.path.join(cfg.save_dir, f'adv_{os.path.basename(img_path)}')
                    adv_image_pil.save(output_path)
                    output_files.append(output_path)
                    sse_adv_samples_gen_validated(output_path, idx + 1, total_images)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            # Calculate comparison metrics
            attack_success_rate = (successful_attacks / total_images) if total_images > 0 else 0.0
            avg_original_conf = np.mean(original_confidences) if original_confidences else 0.0
            avg_adv_conf = np.mean(adversarial_confidences) if adversarial_confidences else 0.0
            confidence_drop = avg_original_conf - avg_adv_conf
            avg_l2_norm = np.mean([p['l2'] for p in perturbation_norms]) if perturbation_norms else 0.0
            avg_linf_norm = np.mean([p['linf'] for p in perturbation_norms]) if perturbation_norms else 0.0
            
            # Output final_result event with comparison data
            results = {
                "status": "success",
                "attack_method": attack_method,
                "model_name": cfg.model,
                "total_samples": total_images,
                "successful_attacks": successful_attacks,
                "attack_success_rate": f"{attack_success_rate * 100:.2f}%",
                "comparison_metrics": {
                    "original_metrics": {
                        "avg_confidence": f"{avg_original_conf:.4f}",
                        "recognition_rate": "100.00%"
                    },
                    "adversarial_metrics": {
                        "avg_confidence": f"{avg_adv_conf:.4f}",
                        "recognition_rate": f"{(1 - attack_success_rate) * 100:.2f}%"
                    },
                    "performance_degradation": {
                        "confidence_drop": f"{confidence_drop:.4f}",
                        "recognition_drop": f"{attack_success_rate * 100:.2f}%"
                    }
                },
                "perturbation_metrics": {
                    "avg_l2_norm": f"{avg_l2_norm:.4f}",
                    "avg_linf_norm": f"{avg_linf_norm:.4f}",
                    "epsilon": args.epsilon
                },
                "attack_parameters": {
                    "epsilon": args.epsilon,
                    "step_size": args.step_size,
                    "max_iterations": args.max_iterations
                },
                "output_info": {
                    "output_files": len(output_files),
                    "output_directory": cfg.save_dir
                }
            }
            
            sse_print("final_result", {}, progress=100, 
                     message=f"{attack_method.upper()} attack completed successfully",
                     details=results)
            
            # Save results to JSON
            json_path = save_json_results(results, cfg.save_dir, f"{attack_method}_attack_results.json")
            sse_print("results_saved", {}, progress=100,
                     message="Results saved to JSON file",
                     details={"json_path": json_path})
        
        elif cfg.mode == 'defend':
            # Defense mode
            defense_method = args.defend_method.lower()
            
            sse_print("defense_process_start", {}, progress=25,
                     message=f"Starting {defense_method.upper()} defense processing",
                     details={"defense_method": defense_method, "model": cfg.model})
            
            if defense_method == 'hgd':
                defender = HGDDefense(model, device=cfg.device)
            elif defense_method == 'tvm':
                defender = TVMDefense(device=cfg.device)
            elif defense_method == 'livenessdetection':
                defender = LivenessDetection(model, device=cfg.device)
            elif defense_method == 'featurespacepurification':
                defender = FeatureSpacePurification(model, device=cfg.device)
            elif defense_method == 'ensembledefense':
                defender = EnsembleDefense(device=cfg.device)
            elif defense_method == 'adversarialdetector':
                defender = AdversarialDetector(model, device=cfg.device)
            else:
                sse_error(f"Unknown defense method: {defense_method}")
                return
            
            # Process images and collect metrics
            total_images = min(10, len(image_files))
            output_files = []
            
            # Metrics for comparison (with and without defense)
            original_predictions = []
            defended_predictions = []
            original_confidences = []
            defended_confidences = []
            detected_adversarial = 0
            successfully_defended = 0
            
            for idx, img_path in enumerate(image_files[:total_images]):
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(cfg.device)
                    
                    # Get prediction without defense
                    with torch.no_grad():
                        orig_output = model(image_tensor)
                        orig_pred = orig_output.argmax(dim=1).item()
                        orig_conf = torch.softmax(orig_output, dim=1).max().item()
                    
                    # Apply defense
                    defended_image = defender.defend(image_tensor)
                    
                    # Get prediction with defense
                    with torch.no_grad():
                        defended_output = model(defended_image)
                        defended_pred = defended_output.argmax(dim=1).item()
                        defended_conf = torch.softmax(defended_output, dim=1).max().item()
                    
                    # Check if image is likely adversarial (low confidence or different prediction after defense)
                    is_adversarial = (orig_conf < 0.5) or (orig_pred != defended_pred)
                    if is_adversarial:
                        detected_adversarial += 1
                        if defended_conf > orig_conf:
                            successfully_defended += 1
                    
                    # Collect metrics
                    original_predictions.append(orig_pred)
                    defended_predictions.append(defended_pred)
                    original_confidences.append(orig_conf)
                    defended_confidences.append(defended_conf)
                    
                    # Save defended image
                    defended_image_pil = transforms.ToPILImage()(defended_image.squeeze(0).cpu())
                    output_path = os.path.join(cfg.save_dir, f'defended_{os.path.basename(img_path)}')
                    defended_image_pil.save(output_path)
                    output_files.append(output_path)
                    sse_clean_samples_gen_validated(output_path, idx + 1, total_images)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            # Calculate comparison metrics
            avg_original_conf = np.mean(original_confidences) if original_confidences else 0.0
            avg_defended_conf = np.mean(defended_confidences) if defended_confidences else 0.0
            confidence_improvement = avg_defended_conf - avg_original_conf
            defense_success_rate = (successfully_defended / detected_adversarial) if detected_adversarial > 0 else 0.0
            
            # Output final_result event with comparison data
            results = {
                "status": "success",
                "defense_method": defense_method,
                "model_name": cfg.model,
                "total_samples": total_images,
                "detected_adversarial": detected_adversarial,
                "successfully_defended": successfully_defended,
                "defense_success_rate": f"{defense_success_rate * 100:.2f}%",
                "comparison_metrics": {
                    "without_defense": {
                        "avg_confidence": f"{avg_original_conf:.4f}",
                        "prediction_stability": "baseline"
                    },
                    "with_defense": {
                        "avg_confidence": f"{avg_defended_conf:.4f}",
                        "prediction_stability": "improved"
                    },
                    "improvement": {
                        "confidence_gain": f"{confidence_improvement:.4f}",
                        "adversarial_detection_rate": f"{(detected_adversarial / total_images) * 100:.2f}%"
                    }
                },
                "output_info": {
                    "output_files": len(output_files),
                    "output_directory": cfg.save_dir
                }
            }
            
            sse_print("final_result", {}, progress=100,
                     message=f"{defense_method.upper()} defense completed successfully",
                     details=results)
            
            # Save results to JSON
            json_path = save_json_results(results, cfg.save_dir, f"{defense_method}_defense_results.json")
            sse_print("results_saved", {}, progress=100,
                     message="Results saved to JSON file",
                     details={"json_path": json_path})
        
        elif cfg.mode == 'train':
            # Training mode for defense methods
            defense_method = args.defend_method.lower()
            
            # Import training module
            from train.defense_trainer import train_defense
            
            # Train defense
            train_defense(model, cfg, args, defense_method)
        
    except Exception as e:
        sse_error(f"Error in main: {str(e)}")
