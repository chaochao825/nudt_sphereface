import torch
import torch.nn as nn
import os
import glob
from PIL import Image
import torchvision.transforms as transforms
from .sphereface_model import SphereFaceModel
from utils.sse import sse_adv_samples_gen_validated, sse_clean_samples_gen_validated, sse_epoch_progress, sse_error

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
            
            for idx, img_path in enumerate(image_files[:total_images]):
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(cfg.device)
                    
                    # Get original prediction
                    with torch.no_grad():
                        orig_output = model(image_tensor)
                        orig_pred = orig_output.argmax(dim=1).item()
                    
                    # Generate adversarial example
                    adv_image = attacker.attack(image_tensor)
                    
                    # Get adversarial prediction
                    with torch.no_grad():
                        adv_output = model(adv_image)
                        adv_pred = adv_output.argmax(dim=1).item()
                    
                    # Check if attack was successful
                    if orig_pred != adv_pred:
                        successful_attacks += 1
                    
                    # Save adversarial image
                    adv_image_pil = transforms.ToPILImage()(adv_image.squeeze(0).cpu())
                    output_path = os.path.join(cfg.save_dir, f'adv_{os.path.basename(img_path)}')
                    adv_image_pil.save(output_path)
                    output_files.append(output_path)
                    sse_adv_samples_gen_validated(output_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            # Output final_result event
            attack_success_rate = (successful_attacks / total_images * 100) if total_images > 0 else 0.0
            from utils.sse import sse_print
            sse_print("final_result", {
                "status": "success",
                "attack_method": attack_method,
                "total_images": total_images,
                "successful_attacks": successful_attacks,
                "attack_success_rate": f"{attack_success_rate:.2f}%",
                "output_files": len(output_files),
                "epsilon": args.epsilon,
                "iterations": args.max_iterations
            })
        
        elif cfg.mode == 'defend':
            # Defense mode
            defense_method = args.defend_method.lower()
            
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
            
            # Process images
            output_files = []
            for img_path in image_files[:min(10, len(image_files))]:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(cfg.device)
                    
                    # Apply defense
                    defended_image = defender.defend(image_tensor)
                    
                    # Save defended image
                    defended_image_pil = transforms.ToPILImage()(defended_image.squeeze(0).cpu())
                    output_path = os.path.join(cfg.save_dir, f'defended_{os.path.basename(img_path)}')
                    defended_image_pil.save(output_path)
                    output_files.append(output_path)
                    sse_clean_samples_gen_validated(output_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            # Output final_result event for defend mode
            from utils.sse import sse_print
            sse_print("final_result", {
                "status": "success",
                "defense_method": defense_method,
                "total_images": min(10, len(image_files)),
                "output_files": len(output_files)
            })
        
        elif cfg.mode == 'train':
            # Training mode for defense methods
            defense_method = args.defend_method.lower()
            
            # Import training module
            from train.defense_trainer import train_defense
            
            # Train defense
            train_defense(model, cfg, args, defense_method)
        
    except Exception as e:
        sse_error(f"Error in main: {str(e)}")
