import subprocess
import time
import numpy as np

"""
输入需要几块gpu，每块gpu多大
在条件满足时立刻输出visible 的写法
"""

def get_CUDA_VISIBLE_DEVICES(queue,GpuRicher = False,returnList = False):
    n = len(queue)
    
    while(1):
        rc, gpuFree = subprocess.getstatusoutput('nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits')
        gpuFree = gpuFree.split('\n')
        gpuFree = np.array(list(map(int, gpuFree)))
        
        for i_th in range(3):
            rc, gpuFree_tmp = subprocess.getstatusoutput('nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits')
            gpuFree_tmp = gpuFree_tmp.split('\n')
            gpuFree_tmp = np.array(list(map(int, gpuFree_tmp)))
            gpuFree = np.minimum(gpuFree,gpuFree_tmp)
            time.sleep(1)
        
        for i in range(n):
            needNumbers,needFree = queue[i][0],queue[i][1]
            idx = np.argwhere(gpuFree>needFree)
            if idx.shape[0] >= needNumbers:                
                idx = idx.transpose(1,0)[0]
                if not GpuRicher:
                    # 使用argsort()函数获取降序排序的下标，从显存最大的卡开始使用
                    idx = np.argsort(-gpuFree)
                    idx = idx[:needNumbers]#防止零碎的显存占用
                res  = (",".join(str(x) for x in idx))
                
                if returnList == True:
                    return idx
                return res
        
        time.sleep(10)#每十秒查询一次
        
    return 0

# 用法
########## auto fetch gpus ##############
# queue = [[1,20000],[2,10000],[4,5500],[8,3000]]#bz 32
# import gpusHelper
# os.environ["CUDA_VISIBLE_DEVICES"] = gpusHelper.get_CUDA_VISIBLE_DEVICES(queue)#当前空闲的GPU
##########################################
# queue = [[2,800]]
# get_CUDA_VISIBLE_DEVICES(queue)