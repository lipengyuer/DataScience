# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html
import scrapy

class HupuItem(scrapy.Item):
    # define the fields for your item here like:
    isAdvPost = scrapy.Field()#是否是主贴
    type = scrapy.Field()
    pageNO = scrapy.Field()
    ifSticky = scrapy.Field()#是否为置顶的跟帖
    url = scrapy.Field()
    post_id = scrapy.Field()
    if_lighted = scrapy.Field()
    floor_num = scrapy.Field()
    post_time = scrapy.Field()
    num_light = scrapy.Field()
    num_gold = scrapy.Field()
    num_gold_pro = scrapy.Field()
    uid = scrapy.Field()
    cited_floor = scrapy.Field()
    cited_uid = scrapy.Field()
    cited_content = scrapy.Field()
    content = scrapy.Field()
    stat_date = scrapy.Field()
    title = scrapy.Field()
    mainBlock = scrapy.Field()
    detailBlock = scrapy.Field()
    num_rec = scrapy.Field()
    ori_context = scrapy.Field()
    uname = scrapy.Field()
    num_reply = scrapy.Field()
    num_light_reposts = scrapy.Field()
    num_brows = scrapy.Field()
