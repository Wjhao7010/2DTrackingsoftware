#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# The 3D-ZeF challenge gt files
ZEF_01="$DIR/train/ZebraFish-01/gt/gt.txt"
ZEF_02="$DIR/train/ZebraFish-02/gt/gt.txt"
ZEF_03="$DIR/train/ZebraFish-03/gt/gt.txt"
ZEF_04="$DIR/train/ZebraFish-04/gt/gt.txt"
# New header for the 3D-ZeF files
ZEF_HEADER="frame,id,3d_x,3d_y,3d_z,camT_x,camT_y,camT_left,camT_top,camT_width,camT_height,camT_occlusion,camF_x,camF_y,camF_left,camF_top,camF_width,camF_height,camF_occlusion"
changeHeader () {
    # The first argument MUST be the header
    HEADER="$1"
    shift
    echo "Adding following header: $HEADER"
    for FILE in "$@"; do
        # Read first line of file
        FIRST_LINE=$(head -n 1 $FILE)
        # Check whether the files include the header, otherwise add it as the first line
        if [ "$FIRST_LINE" = "$HEADER" ]; then
            echo "Header already included"
        else
            sed -i '1s/^/'$HEADER'\n/' $FILE
            echo "Added header to file: $FILE"
        fi
    done
}
changeHeader $ZEF_HEADER $ZEF_01 $ZEF_02 $ZEF_03 $ZEF_04
