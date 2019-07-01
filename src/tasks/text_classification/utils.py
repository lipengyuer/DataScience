"""常用工具"""
import os
import chardet

def find_all_files(dir, file_list):
    if os.path.isfile(dir):
        file_list.append(dir)
        return file_list 
    else:
        dir_list = os.listdir(dir)
        for a_dir in dir_list:
            find_all_files(dir + "/" + a_dir, file_list)
        return file_list

def read_lines_big_file(file_name):
    try:
        f = open(file_name, 'r', encoding='utf8')
        line = f.readline() 
    except Exception as e:
#         print("编码不是utf8,试一下gbk。", e)
        pass
        
    try:
        f = open(file_name, 'r', encoding='gbk')
        line = f.readline() 
    except Exception as e:
        #print("编码也不是gbk,放弃吧朋友。", e)
        return []
        
    while line!="":
        yield line
        #print(line)
        try:
            line = f.readline()
        except:
            pass
    f.close()
    
def read_lines_small_file(file_name):
    flag = True
    lines = []
    if flag==True:
        try:
            f = open(file_name, 'r')
            lines = f.readlines() 
            lines = list(map(lambda x: [x, file_name], lines))
            flag = False
        except Exception as e:
    #         print("编码不是utf8,试一下gbk。", e)
            pass

    if flag==True:
        try:
            f = open(file_name, 'r', encoding='utf8')
            lines = f.readlines() 
            lines = list(map(lambda x: [x, file_name], lines))
            flag = False
        except Exception as e:
            pass
#             print("编码不是utf8,试一下gbk。", e)

    if flag==True:        
        try:
            f = open(file_name, 'r', encoding='gbk')
            lines = f.readlines()
            lines = list(map(lambda x: [x, file_name], lines))
            flag = False
        except Exception as e:
            pass
            #print("编码也不是gbk,放弃吧朋友。", e)
    f.close()
    return lines
    
if __name__ == '__main__':
    a_list = []
    print(a_list)
    res = find_all_files(r'D:\backup\mydata\tieba_posts_voice\school_id=5', [])
    print("asdasdsd")
    print(len(a_list))
    for line in res: 
        print(line)
        asd = read_lines_small_file(line)
        for a in asd:print(a)

            
            
            