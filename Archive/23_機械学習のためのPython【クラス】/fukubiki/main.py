import random 
import time

class Fukubiki():
    def __init__(self, number_of_balls=100, win=2):
        self.NoB = number_of_balls
        self.win = win
        self.lose = self.NoB - self.win
        self.result=[]
        self.balls=[]
        for i in range(self.win):
            self.balls.append("win")
        for i in range(self.lose):
            self.balls.append("lose")
        
    def pick_ball(self, times=1, wait=3):
        if len(self.balls)<times:
            print("Balls are not enough...")
        else:
            for i in range(times):
                time.sleep(wait)
                n=random.randint(0,len(self.balls)-1)
                r=self.balls.pop(n)
                if r=="win":
                    print("You win! Congratulations!")
                else:
                    print("You lose...Sorry...")
                self.result.append(r)
                if len(self.balls)==0:
                    print("All balls are gone.")