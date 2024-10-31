# In the name of God
import argparse
import generate_markers

def main():
    parser = argparse.ArgumentParser("#TODO: add description")

    parser.add_argument(
        "-g", "--generate-markers",
        action="store_true",
        help="Generates and saves aruco markers"
    )
    parser.add_argument(
        "--marker-ids",
        type=int,
        nargs='+',
        help="Specify a list of marker ids"
    )
    parser.add_argument(
        "--file-path",
        type=str,
        help="Specify output file path"
    )
    parser.add_argument(
        "--marker-size",
        type=int,
        default=400,
        help="Specify marker size (in pixels)"
    )
    
    args = parser.parse_args()

    if args.generate_markers:
        generate_markers.generate_markers(
            marker_ids=args.marker_ids,
            file_path=args.file_path,
            marker_size=args.marker_size
        )

if __name__ == "__main__":
    main()