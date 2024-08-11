import zipfile
from argparse import ArgumentParser
import sys
import os

def unzip(path_to_zip_file: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    try:
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        
        print("\x1b[1;32m" + "Extracted to ", os.path.abspath(dest_folder) + '\x1b[0m')
    
    except:
        print('\x1b[1;31m' + "Error extracting ", os.path.abspath(path_to_zip_file) + '\x1b[0m' )



def parse_args():
    parser = ArgumentParser(description="Unzip archive")
    parser.add_argument('file', help='Input file to process.')
    parser.add_argument("folder", type=str, default='.', help="Folder to unzip file")
    return parser.parse_args()



if __name__=="__main__":
    # parse arguments
    try: 
        args = parse_args()
    except Exception as e:
        print('Arguments parsing error.\n' + str(e), file=sys.stderr)

    unzip(args.file, dest_folder=args.folder)