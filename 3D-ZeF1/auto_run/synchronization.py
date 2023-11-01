import paramiko
import sys
import os
import shutil

def getSynFile(base_dir):
    server_file = [
        # ['e:\\3D-ZeF/common/utility.py', '/home/huangjinze/code/3D-ZeF/common/utility.py'],
        # ['e:\\3D-ZeF/modules/detection/BgDetector.py', '/home/huangjinze/code/3D-ZeF/modules/detection/BgDetector.py'],
        # ['e:\\3D-ZeF/threading_bgdetect.py', '/home/huangjinze/code/3D-ZeF/threading_bgdetect.py'],
        # ['e:\\3D-ZeF/auto_run/track.sh', '/home/huangjinze/code/3D-ZeF/auto_run/track.sh'],
        # ['e:\\3D-ZeF/modules/tracking/tools/utilits.py', '/home/huangjinze/code/3D-ZeF/modules/tracking/tools/utilits.py'],
        # ['e:\\3D-ZeF/modules/tracking/SortTracker.py', '/home/huangjinze/code/3D-ZeF/modules/tracking/SortTracker.py'],
        # ['e:\\3D-ZeF/modules/tracking/interpolation.py', '/home/huangjinze/code/3D-ZeF/modules/tracking/interpolation.py'],
        # ['e:\\3D-ZeF/threading_track.py', '/home/huangjinze/code/3D-ZeF/threading_track.py'],
        # ['e:\\3D-ZeF/threading_interpolation.py', '/home/huangjinze/code/3D-ZeF/threading_interpolation.py'],

        ['E:\data/3D_pre/D1_T1/settings.ini', base_dir + 'D1_T1/settings.ini'],
        ['E:\data/3D_pre/D1_T2/settings.ini', base_dir + 'D1_T2/settings.ini'],
        ['E:\data/3D_pre/D1_T3/settings.ini', base_dir + 'D1_T3/settings.ini'],
        ['E:\data/3D_pre/D1_T4/settings.ini', base_dir + 'D1_T4/settings.ini'],
        ['E:\data/3D_pre/D1_T5/settings.ini', base_dir + 'D1_T5/settings.ini'],

        ['E:\data/3D_pre/D2_T1/settings.ini', base_dir + 'D2_T1/settings.ini'],
        ['E:\data/3D_pre/D2_T2/settings.ini', base_dir + 'D2_T2/settings.ini'],
        ['E:\data/3D_pre/D2_T3/settings.ini', base_dir + 'D2_T3/settings.ini'],
        ['E:\data/3D_pre/D2_T4/settings.ini', base_dir + 'D2_T4/settings.ini'],
        ['E:\data/3D_pre/D2_T5/settings.ini', base_dir + 'D2_T5/settings.ini'],

        ['E:\data/3D_pre/D3_T1/settings.ini', base_dir + 'D3_T1/settings.ini'],
        ['E:\data/3D_pre/D3_T2/settings.ini', base_dir + 'D3_T2/settings.ini'],
        ['E:\data/3D_pre/D3_T3/settings.ini', base_dir + 'D3_T3/settings.ini'],
        ['E:\data/3D_pre/D3_T4/settings.ini', base_dir + 'D3_T4/settings.ini'],
        ['E:\data/3D_pre/D3_T5/settings.ini', base_dir + 'D3_T5/settings.ini'],

        ['E:\data/3D_pre/D4_T1/settings.ini', base_dir + 'D4_T1/settings.ini'],
        ['E:\data/3D_pre/D4_T2/settings.ini', base_dir + 'D4_T2/settings.ini'],
        ['E:\data/3D_pre/D4_T3/settings.ini', base_dir + 'D4_T3/settings.ini'],
        ['E:\data/3D_pre/D4_T4/settings.ini', base_dir + 'D4_T4/settings.ini'],
        ['E:\data/3D_pre/D4_T5/settings.ini', base_dir + 'D4_T5/settings.ini'],

        ['E:\data/3D_pre/D5_T1/settings.ini', base_dir + 'D5_T1/settings.ini'],
        ['E:\data/3D_pre/D5_T2/settings.ini', base_dir + 'D5_T2/settings.ini'],
        ['E:\data/3D_pre/D5_T4/settings.ini', base_dir + 'D5_T4/settings.ini'],
        ['E:\data/3D_pre/D5_T5/settings.ini', base_dir + 'D5_T5/settings.ini'],

        ['E:\data/3D_pre/D6_T1/settings.ini', base_dir + 'D6_T1/settings.ini'],
        ['E:\data/3D_pre/D6_T2/settings.ini', base_dir + 'D6_T2/settings.ini'],
        ['E:\data/3D_pre/D6_T4/settings.ini', base_dir + 'D6_T4/settings.ini'],
        ['E:\data/3D_pre/D6_T5/settings.ini', base_dir + 'D6_T5/settings.ini'],

        ['E:\data/3D_pre/D7_T1/settings.ini', base_dir + 'D7_T1/settings.ini'],
        ['E:\data/3D_pre/D7_T2/settings.ini', base_dir + 'D7_T2/settings.ini'],
        ['E:\data/3D_pre/D7_T4/settings.ini', base_dir + 'D7_T4/settings.ini'],
        ['E:\data/3D_pre/D7_T5/settings.ini', base_dir + 'D7_T5/settings.ini'],

        ['E:\data/3D_pre/D8_T2/settings.ini', base_dir + 'D8_T2/settings.ini'],
        ['E:\data/3D_pre/D8_T4/settings.ini', base_dir + 'D8_T4/settings.ini'],

    ]
    return server_file

def getBaseDir(ip):
    if ip == '10.2.151.127':
        base_dir = '/home/huangjinze/code/data/zef/'
    if ip == '10.2.151.128':
        base_dir = '/home/huangjinze/code/data/zef/'
    if ip == '10.2.151.129':
        base_dir = '/home/data/HJZ/zef/'
    return base_dir

def deleteRemoteFile(dt, ip, name, passwd):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())#第一次登录的认证信息
    ssh.connect(hostname=ip, port=22, username=name, password=passwd) # 连接服务器
    stdin, stdout, stderr = ssh.exec_command(f'rm -rf {dt}') # 执行命令
    ssh.close()

def uploadFile2Remote(win_file, linux_file, ip, name, passwd):
    transport = paramiko.Transport((ip, 22))
    transport.connect(username=name, password=passwd)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.put(win_file, linux_file)
    transport.close()

def getPathFromRoot_File(root_dir, file_type):
    file_list = []
    for filepath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(file_type):
                file_list.append(os.path.join(filepath, filename))
    return file_list
if __name__ == '__main__':
    server_info = [
        ['10.2.151.127', 'huangjinze', 'huangjinze33'],
        ['10.2.151.128', 'huangjinze', 'huangjinze33'],
        ['10.2.151.129', 'huangjinze', 'huangjinze33'],
    ]

    # # 同步服务器
    for ip, name, passwd in server_info:
        base_dir = getBaseDir(ip)
        server_file = getSynFile(base_dir)
        for win_file, linux_file in server_file:
            deleteRemoteFile(win_file, ip, name, passwd)
            uploadFile2Remote(win_file, linux_file, ip, name, passwd)

    floder_info = [
        'D1_T1', 'D1_T2', 'D4_T4', 'D4_T5',
        'D2_T1', 'D2_T2', 'D5_T4', 'D5_T5',
        'D3_T1', 'D3_T2', 'D6_T4', 'D6_T5',
        'D1_T4', 'D1_T5', 'D4_T3',
        'D2_T4', 'D2_T5', 'D7_T5',
        'D3_T4', 'D3_T5', 'D8_T2',
        'D4_T2', 'D1_T3',
        'D5_T2', 'D2_T3',
        'D6_T2', 'D3_T3',
        'D7_T1', 'D4_T1',
        'D7_T2', 'D5_T1',
        'D7_T4', 'D6_T1',
        'D8_T4',
    ]
    print(len(floder_info))
    target_path = "F:\\zef"
    # target_path = "F:\\fish_data"
    source_path = "E:\\data\\3D_pre"
    for i in floder_info:
        sourcefile_list = getPathFromRoot_File(f"E:\\data\\3D_pre\\{i}", ".ini")
        for srcfile in sourcefile_list:
            tgt_file = srcfile.replace(source_path, target_path)
            tgt_filepath = os.path.split(tgt_file)[0]
            if not os.path.exists(tgt_filepath):
                os.makedirs(tgt_filepath)
            shutil.copy(srcfile, tgt_file)
        print(f"copy file from {source_path} to {target_path}")