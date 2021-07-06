import argparse

from multi_cropper import MultiCropper

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Image Scan Assistant v1.0")
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('-i', '--image', help='Input image path')
    group.add_argument('-f', '--folder', help='Input folder path')
    parser.add_argument('-o', '--output', help='Set output folder path', default='./')
    parser.add_argument('-c', '--config', help='Config file path', default="./config/config.yaml")
    parser.add_argument('--threads', help='Total threads', default=4, type=int)

    args = parser.parse_args()

    threaded_cropper = MultiCropper(threads=args.threads, config=args.config)
    if args.folder:
        threaded_cropper.crop_image_folder(in_path=args.folder, out_path=args.output)
    elif args.image:
        threaded_cropper.crop_single_image(in_path=args.folder, out_path=args.output)
