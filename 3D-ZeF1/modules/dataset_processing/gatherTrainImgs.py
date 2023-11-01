import pandas as pd
import sys
import os
from pathlib import Path

def splitDF(df,cam):
    """
    Split the dataframe into front and top view ground truth files with a simplified structure
    """
    if cam.lower()=='front':
        prefix = 'camF_'
    elif cam.lower()=='top':
        prefix = 'camT_'
    else:
        print("Couldn't understand the cam input. Closing.")
        sys.exit()

    df = df.rename(columns={'{}x'.format(prefix): 'x',
                            '{}y'.format(prefix): 'y',
                            '{}left'.format(prefix): 'left',
                            '{}top'.format(prefix): 'top',
                            '{}width'.format(prefix): 'width',
                            '{}height'.format(prefix): 'height',
                            '{}occlusion'.format(prefix): 'occlusion'})

    df_out = df[['filename','frame','id','x','y','left','top','width','height','occlusion']]
    return df_out

def createSymlinksToImages(seq_dirs,cam,out_dir):
    """
    Creates an image directory for a given camera view [top, front] and creates symlinks to every image in the specified view of the sequence folders [zebrafish-01, zebrafish-02, zebrafish-03, zebrafish-04]
    """
    if cam.lower()=='front':
        imgdir = 'img2'
    elif cam.lower()=='top':
        imgdir = 'img1'
    else:
        print("Couldn't understand the cam input. Closing.")
        sys.exit()

    try:
        out_dir.mkdir(parents=True)
    except FileExistsError:
        print("WARNING: Directory {} already exists.".format(out_dir))
        while True:
            try:
                response = str(input("Continue? [y]/n\n") or "y")
            except ValueError:
                print("Try again, please.")
                continue
            if response.lower() in ["n","no"]:
                print("Closing.")
                sys.exit()
            elif response.lower() in ["y","yes"]:
                break
            else:
                print("Didn't understand. Try again, please.")
                continue

    out_dir = str(out_dir.resolve())
    for seq in seq_dirs:
        seq_imgdir = Path(seq,imgdir)
        for f in os.listdir(seq_imgdir):
            if f.endswith('.jpg'):
                from_ = str(Path(seq_imgdir,f).resolve())
                to_ = Path(out_dir,"zef-{}_{}.jpg".format(str(seq)[-2:],f[:-4]))
                try:
                    os.symlink(from_,to_)
                except:
                    print("Symlink from {} to {} already exists. Continuing.".format(from_,to_))
                    pass

def run():
    train_dir = Path('/home/data/HJZ/3DZeF20/train')
    all_dir = Path(train_dir,'all')
    front_dir = Path(all_dir,'front')
    top_dir = Path(all_dir,'top')

    try:
        all_dir.mkdir(parents=True)
    except FileExistsError:
        print("WARNING: Directory {} already exists.".format(all_dir))
        while True:
            try:
                response = str(input("Continue? [y]/n\n") or "y")
            except ValueError:
                print("Try again, please.")
                continue
            if response.lower() in ["n","no"]:
                print("Closing.")
                sys.exit()
            elif response.lower() in ["y","yes"]:
                break
            else:
                print("Didn't understand. Try again, please.")
                continue

    # Create dirs for top and front images
    front_dir.mkdir(parents=True, exist_ok=True)
    top_dir.mkdir(parents=True, exist_ok=True)

    # Read groundtruth files
    seqs = ['ZebraFish-01','ZebraFish-02','ZebraFish-03','ZebraFish-04']
    seq_dirs = [seq for seq in train_dir.iterdir() if seq.name in seqs]

    gt_dfs = {seq.name: pd.read_csv(Path(seq,'gt','gt.txt')) for seq in seq_dirs}

    # Add filename to all annotations
    for key in gt_dfs:
        df = gt_dfs[key]
        seq_no = key[-2:] # sequence number
        filenames = ["zef-{}_{}.jpg".format(seq_no,str(f).zfill(6)) for f in df['frame'].values]
        gt_dfs[key]['filename'] = filenames

    # Gather the ground truth dataframes into a single dataframe
    df_all = pd.concat([gt_dfs[key] for key in gt_dfs], ignore_index=True)
    del gt_dfs # we don't need the dict anymore

    # Split the dataframe into a top and front ground truth csv and place them in their respective directories
    splitDF(df_all,cam='front').to_csv(Path(front_dir,'gt.txt'),index=False)
    splitDF(df_all,cam='top').to_csv(Path(top_dir,'gt.txt'),index=False)

    # Create symlinks for all the images
    imgs_front = createSymlinksToImages(seq_dirs,cam='front',out_dir=Path(front_dir,'imgs'))
    imgs_top = createSymlinksToImages(seq_dirs,cam='top',out_dir=Path(top_dir,'imgs'))


if __name__ == '__main__':
    """
    Used for gathering all the images and structuring the annotation files for easier training of new detectors
    """
    run()
