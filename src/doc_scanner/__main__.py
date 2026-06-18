import cv2
import argparse
from pathlib import Path
from doc_scanner.core.detector import detect_document
from doc_scanner.core.enhancement import enhance_image
from doc_scanner.core.process_text import binarize_for_text
from doc_scanner.utils import filters
from doc_scanner import config
from doc_scanner.core.OCR_evaluation import ocr, text_accuracy, image_metrics
from doc_scanner.core.OCR_evaluation import plot_image

def main():
    parser = argparse.ArgumentParser(description="Document Scanner CLI (Advanced Pipeline)")
    parser.add_argument("--input", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, help="Path to save output image")

    # Enhancement toggles
    parser.add_argument("--enhance", action="store_true", help="Apply image enhancement (shadow removal, sharpening)")
    parser.add_argument("--no-shadow-remove", action="store_true", help="Disable background division shadow removal")
    parser.add_argument("--no-deskew", action="store_true", help="Disable automatic skew angle correction")
    parser.add_argument("--no-orient", action="store_true", help="Disable automatic text orientation detection")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Manual rotation angle (clockwise)")

    # Text Binarization options
    parser.add_argument("--text", action="store_true", help="Apply text binarization")
    parser.add_argument("--method", type=str, choices=["sauvola", "adaptive", "otsu"], default="sauvola", help="Binarization method")
    parser.add_argument("--sauvola-k", type=float, default=0.15, help="K parameter for Sauvola binarization (typical: 0.1 to 0.3)")
    parser.add_argument("--sauvola-window", type=int, default=0, help="Window size for Sauvola (odd integer, 0 for auto)")
    parser.add_argument("--aspect-ratio", type=str, choices=["none", "A4", "letter"], default="none", help="Force output to standard aspect ratio")

    parser.add_argument("--debug", action="store_true", help="Show debug windows")
    args = parser.parse_args()

    # Default values if not provided
    input_path = Path(args.input) if args.input else config.INPUT_DIR / "test6.jpg"
    output_dir = Path(args.output).parent if args.output else config.OUTPUT_DIR
    output_name = Path(args.output).name if args.output else "result.jpg"

    print(f"Scanning document: {input_path}")
    try:
        # 1. Detection & Warping
        result = detect_document(str(input_path))
        final_image = result.warped

        # 2. Enhancement
        # Note: If --enhance is passed, we apply full enhancements.
        # But we also run auto-orient and deskew by default unless disabled.
        remove_shadows = not args.no_shadow_remove
        auto_orient = not args.no_orient
        deskew = not args.no_deskew

        # We run basic rotation/deskew on the warped image even if --enhance is not specified,
        # since aligning orientation and fixing perspective errors is critical.
        if auto_orient:
            print("Detecting orientation...")
            final_image, rot_angle = filters.auto_rotate_text(final_image)
            if rot_angle > 0:
                print(f"Auto-rotated text by {rot_angle} degrees clockwise.")

        if args.rotate > 0:
            print(f"Applying manual rotation of {args.rotate} degrees...")
            if args.rotate == 90:
                final_image = cv2.rotate(final_image, cv2.ROTATE_90_CLOCKWISE)
            elif args.rotate == 180:
                final_image = cv2.rotate(final_image, cv2.ROTATE_180)
            elif args.rotate == 270:
                final_image = cv2.rotate(final_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if deskew:
            print("Estimating skew...")
            skew_angle = filters.estimate_skew_angle(final_image)
            if abs(skew_angle) > 0.1:
                print(f"Correcting text skew: {skew_angle:.2f} degrees")
                final_image = filters.deskew(final_image, skew_angle)

        if args.enhance:
            print("Enhancing image (shadow removal, CLAHE, sharpening)...")
            final_image = enhance_image(
                final_image,
                use_sharpen=True,
                use_unsharp=True,
                use_clahe=not args.text,
                remove_shadows=remove_shadows,
                auto_orient=False, # Already handled above
                deskew=False       # Already handled above
            )
        elif remove_shadows and not args.text:
            # If not enhancing, but shadow removal is not disabled, let's apply shadow removal
            # to make the color/gray scan beautiful
            print("Removing shadows...")
            final_image = filters.remove_shadows(final_image)

        # 3. Text Processing
        if args.text:
            print(f"Processing text (Binarization: {args.method})...")
            if args.method == "sauvola":
                w_size = args.sauvola_window
                if w_size <= 0:
                    h, w = final_image.shape[:2]
                    w_size = int(max(h, w) / 45)
                    if w_size % 2 == 0:
                        w_size += 1
                    w_size = max(5, w_size)
                print(f"Applying Sauvola (k={args.sauvola_k}, window_size={w_size})...")
                final_image = filters.sauvola_threshold(final_image, window_size=w_size, k=args.sauvola_k)
                # Clean binarized output
                final_image = filters.apply_morphology(final_image, kernel_size=(2, 2), op=cv2.MORPH_OPEN)
            elif args.method == "adaptive":
                final_image = binarize_for_text(final_image, "adaptive")
                final_image = filters.apply_morphology(final_image, kernel_size=(2, 2), op=cv2.MORPH_OPEN)
            elif args.method == "otsu":
                final_image = filters.otsu_threshold(final_image)

        # 4. Aspect Ratio Correction
        if args.aspect_ratio != "none":
            h, w = final_image.shape[:2]
            is_portrait = h > w

            if args.aspect_ratio == "A4":
                target_ratio = 1.414
            else: # Letter
                target_ratio = 11.0 / 8.5 # 1.294

            if is_portrait:
                target_w = w
                target_h = int(w * target_ratio)
            else:
                target_h = h
                target_w = int(h * target_ratio)

            print(f"Resizing output to standard {args.aspect_ratio} aspect ratio ({target_w}x{target_h})...")
            final_image = cv2.resize(final_image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Save results
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), final_image)
        print(f"Saved result to: {output_path}")

        if args.debug:
            # Show stages
            debug_img = result.resized.copy()
            cv2.drawContours(debug_img, [result.contour], -1, (0, 255, 0), 3)

            cv2.imshow("1. Original (Resized)", result.resized)
            cv2.imshow("2. Edges", result.edged)
            cv2.imshow("3. Detected Contour", debug_img)
            cv2.imshow("4. Warped Result", result.warped)
            cv2.imshow("5. Final Processed", final_image)

            print("Press any key to close debug windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # OCR + acc
        img_ocr = cv2.imread(f"{output_path}", cv2.IMREAD_GRAYSCALE)
        output_text = ocr(img_ocr)
        
        img_ocr = cv2.imread(f"{input_path}", cv2.IMREAD_GRAYSCALE)
        input_text = ocr(img_ocr)
        
        _ = detect_document(str(input_path))
        image = _.warped
        img_ocr = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
        A_text = ocr(img_ocr)
        
        st = f"{input_path}".split('\\')
        test = st[-1][:-4]
        
        with open(f"data/eval/ground_truth/{test}.txt", "r", encoding="utf-8") as f:
            ground_truth = f.read()
            
        output_acc = text_accuracy(output_text, ground_truth)
        input_acc = text_accuracy(input_text, ground_truth)
        A_acc = text_accuracy(A_text, ground_truth)
        print(f'output_acc = {output_acc}')
        print(f'input_acc = {input_acc}')
        print(f'A_acc = {A_acc}')
        
        
        # psnr + ssim
        im = image_metrics(image, final_image)
        print(im)
        with open(f"data/eval/metrics/{test}.txt", "w", encoding="utf-8") as file:
            file.write(f"psnr: {im['psnr']}\n")
            file.write(f"ssim: {im['ssim']}")
        
        stages = {
            "1. Original": cv2.imread(f"{input_path}"),
            "2. Ours": cv2.imread(f"{output_path}")
        }
        
        acc_text = [
            f'original_acc = {input_acc}', 
            f'ours_acc = {output_acc}'
        ]
        
        plot_image(stages, acc_text, f"data/eval/plot/{test}.jpg")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
