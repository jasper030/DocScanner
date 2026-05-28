import cv2
import argparse
from pathlib import Path
from doc_scanner.core.detector import detect_document
from doc_scanner import config

def main():
    parser = argparse.ArgumentParser(description="Document Scanner CLI")
    parser.add_argument("--input", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, help="Path to save output image")
    parser.add_argument("--debug", action="store_true", help="Show debug windows")

    args = parser.parse_args()

    # Default values if not provided
    input_path = Path(args.input) if args.input else config.INPUT_DIR / "test6.jpg"
    output_path = Path(args.output) if args.output else config.OUTPUT_DIR / "warped_result.jpg"

    print(f"Scanning document: {input_path}")

    try:
        resized, edged, contour, warped = detect_document(input_path)

        # Save results
        cv2.imwrite(str(output_path), warped)
        print(f"Saved result to: {output_path}")

        if args.debug:
            debug_img = resized.copy()
            cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 3)

            cv2.imshow("Original (Resized)", resized)
            cv2.imshow("Edges", edged)
            cv2.imshow("Contour", debug_img)
            cv2.imshow("Warped", warped)

            print("Press any key to close debug windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
