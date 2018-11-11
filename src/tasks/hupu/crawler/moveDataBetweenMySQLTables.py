# -*- coding: utf-8 -*-
import pymysql

conn_remote = pymysql.connect(host='192.168.1.198',
                                 database="test",
                                 user="root",
                                 password="1q2w3e4r",
                                 charset="utf8")
# conn_remote.execute('set max_allowed_packet=67108864')
cursor1 = conn_remote.cursor()


def insertLines(lines, tableB, str_sql_target):
    cursor1.executemany(str_sql_target, lines)
    conn_remote.commit()
   # cursor1.close()
   # conn_remote.close()
    #print("sdfsfd")

def moveDataFromTabmeA2TableB(tableA, tableB, insertBatchSize=100, fieldNum = 0):
    conn_local = pymysql.connect(host='192.168.1.199',
                                 database="test",
                                 user="root",
                                 password="1q2w3e4r",
                                 charset="utf8")
    cursor = conn_local.cursor(pymysql.cursors.SSCursor)
    cursor.execute('SET NET_WRITE_TIMEOUT = 1000')

    str_sql_src = ""
    str_sql_src += "select * from " + tableA
    
    ss = ["%s" for _ in range(fieldNum)]
    str_sql_target = ""
    str_sql_target += "insert into " + tableB + " values(" + ",".join(ss) + ")"
    
    cursor.execute(str_sql_src)
    count = 0
    temp=[]
    for line in cursor:
        try:
            count += 1
            print(count)
            if line==None:
                continue
            if len(line[-4])>100:
                print(line[-4])
                line[-4] = line[-4].split("\"")[3]
                temp.append(line)
                open("log.txt", 'a+').write(str(line[-4]) + "\n")
            else:
                temp.append(line)
            #if count<1483400:
            #    continue

            #print(line)
            if len(temp)==insertBatchSize:
                insertLines(temp,tableB, str_sql_target)
                temp = []
            #print(line, len(line))
        except:
            for a in temp:
                    print(a)
               # try:
                    insertLines([a], tableB, str_sql_target)
               # except:
               #     open("log.txt",'a+').write(str(count) + "\n")
            temp = []
    insertLines(temp,tableB, str_sql_target)
    cursor.close()
    conn_local.close()


def moveDataFromTabmeA2Local(tableA, fileName):
    conn_local = pymysql.connect(host='192.168.1.198',
                                 database="test",
                                 user="root",
                                 password="1q2w3e4r",
                                 charset="utf8")
    cursor = conn_local.cursor(pymysql.cursors.SSCursor)
    cursor.execute('SET NET_WRITE_TIMEOUT = 1000')

    str_sql_src = ""
    str_sql_src += 'select * from ' + tableA
    print(str_sql_src)
    cursor.execute(str_sql_src)
    count = 0
    #temp = []
    f = open(fileName, "a+",encoding='utf8')
    #import chardet
    temp = []
    for line in cursor:
        count+=1
        print(count)
        if len(temp)==100:
            f.writelines(temp)
            temp = []
            counter = 0
        line = map(lambda x:str(x).replace("#", ""), line)
        line = "#".join(line).encode("utf8")+ b"\n"
        line = line.decode("utf8")
        temp.append(line)
        #print(chardet.detect(line), type(line))
        #f.write(line.decode("utf8"))
    f.writelines(temp)
    f.close()
    cursor.close()
    conn_local.close()

#moveDataFromTabmeA2TableB("hupu_bxj_advocate_posts", "hupu_bxj_advocate_posts",insertBatchSize=1000, fieldNum = 13)
#moveDataFromTabmeA2TableB("hupu_bxj_foll_posts", "hupu_bxj_foll_posts",insertBatchSize=500, fieldNum = 13)
moveDataFromTabmeA2Local("hupu_bxj_advocate_posts_1", "G:\\data\\hupu\\hupu_bxj_advocate_posts.txt")
moveDataFromTabmeA2Local("hupu_bxj_foll_posts_1", "G:\\data\\hupu\\hupu_bxj_foll_posts.txt")


