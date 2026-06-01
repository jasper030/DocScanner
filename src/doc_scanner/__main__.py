import cv2
import argparse
from pathlib import Path
from doc_scanner.core.detector import detect_document
from doc_scanner.core.enhancement import enhance_image
from doc_scanner.core.process_text import process_text_image
from doc_scanner import config

def main():
    parser = argparse.ArgumentParser(description="Document Scanner CLI")
    parser.add_argument("--input", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, help="Path to save output image")
    parser.add_argument("--enhance", action="store_true", help="Apply image enhancement")
    parser.add_argument("--text", action="store_true", help="Apply text binarization")
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

        # 2. Enhancement (Optional)
        if args.enhance:
            print("Enhancing image...")
            final_image = enhance_image(final_image, use_sharpen=True)

        # 3. Text Processing (Optional)
        if args.text:
            print("Processing text...")
            text_result = process_text_image(final_image)
            final_image = text_result.cleaned

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
            
            if args.enhance:
                cv2.imshow("5. Enhanced", final_image if not args.text else enhance_image(result.warped))
            
            if args.text:
                cv2.imshow("6. Text (Cleaned)", final_image)

            print("Press any key to close debug windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
