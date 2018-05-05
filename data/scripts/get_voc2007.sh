#!/bin/bash

start=`date +%s`

# handle optional download dir
if [ -z "$1" ]
  then
    # navigate to ~/data
    echo "navigating to data directory" 
    cd ../
  else
    # check if is valid directory
    if [ ! -d $1 ]; then
        echo $1 "is not a valid directory"
        exit 0
    fi
    echo "navigating to" $1 "..."
    cd $1
fi

echo "Downloading VOC2007 trainval ..."
# Download the data.
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
echo "Downloading VOC2007 test data ..."
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
echo "Done downloading."

# Extract data
echo "Extracting trainval ..."
tar -xvf VOCtrainval_06-Nov-2007.tar
echo "Extracting test ..."
tar -xvf VOCtest_06-Nov-2007.tar
echo "removing tars ..."
rm VOCtrainval_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"