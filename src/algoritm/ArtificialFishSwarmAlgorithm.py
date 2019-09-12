'''
Created on 2019年9月11日

@author: Administrator
'''
#人工鱼群算法
import numpy as np
#第一阶段的人工鱼群只有追寻食物的机制。只让人工鱼寻找食物浓度最高的位置，就可以实现对一些的损失函数的优化了。
class AFish():
    
    def __init__(self):
        self.location = None#描述鱼在参数空间中位置的向量
        self.current_food_density = None
        
class AFSA():
    
    def __init__(self, fish_num = 100, location_dim=2, visual=0.01, try_num_searching_food=3):
        self.location_dim = location_dim
        self.fishes = None
        self.bulletin_fish = None
        self.visual = visual#所有鱼的视野范围。可以为每条鱼设置不同的视野大小，模拟个体差异
        self.try_num_searching_food = try_num_searching_food#多试几次有助于找到更好的方向。
        self.max_fish_in_vision = 10#视野内鱼的最大个数，用于限制鱼的密度
        self.fish_swarms = {}
        self.create_fishes(fish_num, location_dim)

        
    def food_density(self, location):
#         print('location', location[0])
        score = -(np.sum(location*location) + 2*location[1])#z=x**2+y**2 +3.105*y- 1,需要加一个符号，让函数是凸的。如果需要优化的目标函数比较复杂，就没有这么直观了。
        return score
    
    def distance(self, location1, location2):
        return np.dot(location1, location2)
    
    #生成一个步长，目标是视野范围内的随机一个点
    def generate_a_step(self):
        step_vector = np.random.uniform(-self.visual, self.visual, self.location_dim)
        return step_vector
    
    def create_fishes(self, fish_num, location_dim):
        self.fishes = []
        for _ in range(fish_num):
            a_fish = AFish()
            a_fish.location = np.random.random(location_dim) 
#             print("a_fish.location", a_fish.location)
            a_fish.current_food_density = self.food_density(a_fish.location)
            self.fishes.append(a_fish)
            if self.bulletin_fish==None:
                self.bulletin_fish = a_fish
            else:
                if a_fish.current_food_density > self.bulletin_fish.current_food_density:
                    self.bulletin_fish = a_fish
    
    def random_move(self, fish):
        new_location = fish.location + self.generate_a_step()
        new_density = self.food_density(new_location)
        fish.location = new_location
        fish.current_food_density = new_density
               
    def search_food(self, fish):
        for _ in range(self.try_num_searching_food):
            new_location = fish.location + self.generate_a_step()
            new_density = self.food_density(new_location)
            concentration = self.food_density(fish.location)
            if  new_density > concentration:
                fish.location = new_location
                concentration = new_density
            if concentration > self.bulletin_fish.current_food_density:
                self.bulletin_fish = fish
    
    
    def swarm(self, fish):
        fish_location_in_vision = []
        for a_fish in self.fishes:
            if self.distance(fish.location, a_fish.location) < self.visual:
                fish_location_in_vision.append(a_fish.location)
        if 0 < len(fish_location_in_vision) < self.max_fish_in_vision:
            center = np.mean(fish_location_in_vision)
            a_step_to_center = center - fish.location
            fish.location = fish.location + a_step_to_center*np.random.uniform(0, 1)
            #fish.location = (1 - np.random.uniform(0, 1)) + center
            fish.current_food_density = self.food_density(fish.location)
    
    #追尾行为。这里重复计算，一遍展示过程
    def follow(self, fish):
        fatest_fish_in_vision = None
        fish_num_in_vision = 0
        for a_fish in self.fishes:
            if self.distance(a_fish.location, fish.location)<self.visual:
                fish_num_in_vision += 1
                if fatest_fish_in_vision==None or self.food_density(a_fish.location) > self.food_density(fatest_fish_in_vision.location):
                    fatest_fish_in_vision = a_fish
        if fatest_fish_in_vision!=None and fish_num_in_vision<self.max_fish_in_vision:
            a_step_to_fatest = fatest_fish_in_vision.location - fish.location
            fish.location = fish.location + a_step_to_fatest*np.random.uniform(0, 1)
            #fish.location = (1 - np.random.uniform(0, 1)) + center
            fish.current_food_density = self.food_density(fish.location)  
             
    #更新一条鱼的状态，并更新公示板        
    def update_a_fish(self, fish):
        self.search_food(fish)#模拟一条鱼找食物的动作
        self.swarm(fish)
        self.follow(fish)
        self.random_move(fish)#假如鱼没有觅食、追尾或者聚群，就随机移动一下。
        
    def fit(self ,epoch_num=1000):
        for epoch in range(epoch_num):
            for fish in self.fishes:
                self.update_a_fish(fish)
            print("轮次是", epoch)
            print(self.bulletin_fish.location)
#             break
                
    
if __name__ == '__main__':
    afsa = AFSA()
    afsa.fit()
        