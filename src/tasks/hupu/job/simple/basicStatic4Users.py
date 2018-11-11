#统计用户的发帖总数，发帖被推荐的次数，发帖的金币总数，发帖的回复总数，回复被点亮的总数，发帖被浏览的总数；
#发帖的板块分布
#用户跟帖（回复）的总数，跟帖被点亮的总数，跟帖的金币总数，高亮跟帖的数量，高亮跟帖的被亮次数， 跟帖被引用的次数，
#从mysql中获取用户的主贴数据
from hupu.analisys.dbconnection.getMySQL import getConnection
import time
mysqlcon = getConnection()
mysqlcon.setConnecter(host='192.168.1.198', port=3306, user="root",
                      passwd='1q2w3e4r', db='test',
                      use_unicode=True, charset="utf8")

def getAdvocatePosts(uid):
    t1 = time.time()
    sql_str = "select * from hupu_bxj_advocate_posts_1 where uid='" + str(uid) + "'"
    data = mysqlcon.queryWithReturn(sql_str)
    t2 = time.time()
    print("耗时是", t2-t1)
    return data

#统计主贴的基本情况
#统计用户的发帖总数，发帖被推荐的次数，发帖的金币总数，发帖的回复总数，被点亮的回复总数，发帖被浏览的总数；
def countAPosts(posts):
    #对帖子去重，用post_和title
    distinct_posts = {}
    for post in posts:
        key = post[0] + "_" + post[1]
        if key not in distinct_posts:
            distinct_posts[key] = post

    res = {"advocate_post_number": len(distinct_posts.keys()), 'recommend_number': 0, 'gold_number': 0, "reply_number": 0,
           "reply_lighted_number": 0, "browsed_number": 0, "foot_print_main_block": set({}),
           'foot_print_detail': set({}), "basketB_index":0}
    #basketB_index表示一个用户是篮球粉丝的程度，用他在帖子里提到“篮球”的次数来表示。
    for _, post in distinct_posts.items():
        res['recommend_number'] += int(post[7])
        res['gold_number'] += int(post[5])
        res['reply_number'] += int(post[11])
        res['reply_lighted_number'] += int(post[12])
        res['browsed_number'] += int(post[13])
        res['foot_print_main_block'].add(post[2])
        res['foot_print_detail'].add(post[3])

    return res

#从mysql获取跟帖数据
def getFollowPosts(uid):
    t1 = time.time()
    sql_str = "select * from hupu_bxj_foll_posts_1 where uid='" + str(uid) + "'"
    data = mysqlcon.queryWithReturn(sql_str)
    t2 = time.time()
    print("耗时是", t2-t1)
    return data

#统计跟帖的基本情况
#用户跟帖（回复）的总数，跟帖被点亮的总数，跟帖的金币总数，高亮跟帖的数量，高亮跟帖的被亮次数， 跟帖被引用的次数
def countFPosts(posts):
    posts = map(lambda x: x[:-2], posts)
    #对帖子去重，用帖子的post_id和楼层
    postsMap = {}
    for post in posts:
        key = post[0] + "_" + post[2]
        if key not in postsMap:
            postsMap[key] = post

    res = {"follow_post_number": len(postsMap.keys()), 'follow_post_gold_number': 0, 'follow_post_cited_number': 0,
           "follow_post_high_lighted_number": 0, 'followed_post_id':[]}

    for _, post in postsMap.items():
        res['follow_post_gold_number'] += int(post[5])
        res['follow_post_cited_number'] += 1 if post[8]!=0 else 0
        res['follow_post_high_lighted_number'] += post[1]
        res['followed_post_id'].append(post[0])
    return res

if __name__ == '__main__':
    uid = '3'
    aPosts = getAdvocatePosts(uid)
    aStatistics = countAPosts(aPosts)
    fPosts = getFollowPosts(uid)
    fStatistics = countFPosts(fPosts)
    print(aStatistics)
    print(fStatistics)

