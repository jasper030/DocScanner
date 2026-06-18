import cv2
import pytesseract
import Levenshtein
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def ocr(image, lang: str = 'eng', psm: int = 3, oem: int = 3) -> str: 
    if image is None or image.size == 0:
        print("[Warning] ocr: no image")
        return ""
    
    try:
        custom_config = f'--psm {psm} --oem {oem}'
        recognized_text = pytesseract.image_to_string(
            image, 
            lang=lang, 
            config=custom_config
        )
        recognized_text = str(recognized_text)
        return recognized_text.strip()
        
    except pytesseract.TesseractNotFoundError:
        print("[Error] ocr: Tesseract OCR not found.")
        return ""
    except Exception as e:
        print(f"[Error] ocr: unknown error {str(e)}")
        return ""

def text_accuracy(pred_text: str, ground_truth: str) -> float:
    pred_clean = pred_text
    gt_clean = ground_truth

    edit_distance = Levenshtein.distance(pred_clean, gt_clean)
    total_chars = len(gt_clean)
    cer = edit_distance / total_chars
    accuracy = max(0.0, 1.0 - cer) * 100.0
    
    return round(accuracy, 2)

def image_metrics(img_original, img_processed):
    if len(img_original.shape) == 3:
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    if len(img_processed.shape) == 3:
        img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
        
    if img_original.shape != img_processed.shape:
        img_processed = cv2.resize(img_processed, (img_original.shape[1], img_original.shape[0]))

    current_psnr = psnr(img_original, img_processed, data_range=255)
    current_ssim = ssim(img_original, img_processed, data_range=255)
    
    return {
        "psnr": round(current_psnr, 2),
        "ssim": round(current_ssim, 4)
    }


def plot_image(stages_dict: dict, text_dict: list, save_path: str = ""):
    num_stages = len(stages_dict)
    if num_stages == 0:
        return

    fig, axes = plt.subplots(1, num_stages, figsize=(num_stages * 4, 5))
    
    if num_stages == 1:
        axes = [axes]
    
    i = 0
    
    for title, img in stages_dict.items():
        acc_text = text_dict[i]
        ax = axes[i]
        if img is None or img.size == 0:
            ax.text(0.5, 0.5, "Empty Image", ha='center', va='center')
            ax.set_title(title)
            ax.axis('off')
            continue
            
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
        else:
            ax.imshow(img, cmap='gray')
            
        ax.set_title(title + '\n' + acc_text, fontsize=12, fontweight='bold')
        # ax.set_xlabel(acc_text, fontsize=10, color='blue', labelpad=8)
        ax.axis('off')
        
        i += 1
        

    plt.tight_layout()
    if len(save_path):
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
