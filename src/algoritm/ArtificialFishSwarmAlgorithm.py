'''
Created on 2019年9月11日

@author: Administrator
'''
#人工鱼群算法
import numpy as np

class AFish():
    
    def __init__(self):
        self.location = None#描述鱼在参数空间中位置的向量
        self.current_food_concentration = None
        
class AFSA():
    
    def __init__(self, fish_num = 100, location_dim=2, visual=0.01):
        self.location_dim = location_dim
        self.fishes = None
        self.bulletin_fish = None
        self.visual = visual#所有鱼的视野范围。可以为每条鱼设置不同的视野大小，模拟个体差异
        self.create_fishes(fish_num, location_dim)

        
    def food_concentration(self, location):
        score = -np.sum(location*location)#z=x**2+y**2,需要加一个符号，让函数是凸的。如果需要优化的目标函数比较复杂，就没有这么直观了。
        return score
    
    #生成一个步长，目标是视野范围内的随机一个点
    def generate_a_step(self):
        step_vector = np.random.uniform(-self.visual, self.visual, self.location_dim)
        return step_vector
    
    def create_fishes(self, fish_num, location_dim):
        self.fishes = []
        for _ in range(fish_num):
            a_fish = AFish()
            a_fish.location = np.random.random(location_dim) 
            a_fish.current_food_concentration = self.food_concentration(a_fish.location)
            self.fishes.append(a_fish)
            if self.bulletin_fish==None:
                self.bulletin_fish = a_fish
            else:
                if a_fish.current_food_concentration > self.bulletin_fish.current_food_concentration:
                    self.bulletin_fish = a_fish
        
    def update_a_fish(self, fish):
        new_location = fish.location + self.generate_a_step()
        new_concentration = self.food_concentration(new_location)
        concentration = self.food_concentration(fish.location)
        if  new_concentration > concentration:
            fish.location = new_location
            concentration = new_concentration
        if concentration > self.bulletin_fish.current_food_concentration:
            self.bulletin_fish = fish
        
    def fit(self ,epoch_num=1000):
        for epoch in range(epoch_num):
            for fish in self.fishes:
                self.update_a_fish(fish)
            print("轮次是", epoch)
            print(self.bulletin_fish.location)
                
    
if __name__ == '__main__':
    afsa = AFSA()
    afsa.fit()
        