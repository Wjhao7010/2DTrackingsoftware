set /P dirPath=Enter directory path: 

echo %dirPath%

cd detection 
D:\program\Anaconda3\envs\zebrafish\python BgDetector.py -i -f %dirPath% -c 1
D:\program\Anaconda3\envs\zebrafish\python BgDetector.py -i -f %dirPath% -c 2

cd ..
cd tracking

D:\program\Anaconda3\envs\zebrafish\python TrackerVisual.py -i -f %dirPath% -c 1 --preDetector
D:\program\Anaconda3\envs\zebrafish\python TrackerVisual.py -i -f %dirPath% -c 2 --preDetector

cd ..
cd reconstruction

D:\program\Anaconda3\envs\zebrafish\python JsonToCamera.py -f %dirPath% -c 1
D:\program\Anaconda3\envs\zebrafish\python JsonToCamera.py -f %dirPath% -c 2
D:\program\Anaconda3\envs\zebrafish\python TrackletMatching.py -f %dirPath%
D:\program\Anaconda3\envs\zebrafish\python FinalizeTracks.py -f %dirPath%

cd ..

pause