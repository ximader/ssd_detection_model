import os
import requests
import sys 
from argparse import ArgumentParser




def download(url: str, dest_folder: str):
    """ Download file from url """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist
    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename).replace('\\', '/')
    file_path = os.path.abspath(file_path).replace('\\', '/')

    if os.path.isfile(file_path):
        print("\x1b[1;32m" + f"File already downloaded" + '\x1b[0m')
        return 0

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", file_path)
        print()
        chunk_count = 0
        chunk_size = 1024*8
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:

                    # Clear the last line of output
                    CURSOR_UP_ONE = '\x1b[1A'
                    ERASE_LINE = '\x1b[2K'
                    sys.stdout.write(CURSOR_UP_ONE)
                    sys.stdout.write(ERASE_LINE)

                    chunk_count+=1
                    print(f'Received {chunk_count * chunk_size / (1024*1024):.2f} Mb')
                    
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
        
        print("\x1b[1;32m" + "Downloaded to ", os.path.abspath(file_path) + '\x1b[0m')

    else:  # HTTP status code 4XX/5XX
        print('\x1b[1;31m' + "Download failed: status code {}\n{}".format(r.status_code, r.text) + '\x1b[0m' )




def parse_args():
    parser = ArgumentParser(description="Download file from URL")
    parser.add_argument('url', help='Input file to process.')
    parser.add_argument("folder", type=str, default='.', help="Folder to save file")
    return parser.parse_args()




if __name__=="__main__":
    # parse arguments
    try: 
        args = parse_args()
    except Exception as e:
        print('Arguments parsing error.\n' + str(e), file=sys.stderr)

    download(args.url, dest_folder=args.folder)
   