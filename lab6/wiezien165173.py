import numpy as np


class Wiezien165173:
    def __init__(self):
        self.enemyDecisions = [0]
        self.myDecisions = [0]
        self.myPenalty = np.array([[1, 5,  2,  3],
                                   [3, 2,  1,  5],
                                   [4, 7, -6,  1],
                                   [6, 5,  3, -3]])

        self.enemyPenalty = np.transpose(np.array([[1, 5,  2,  3],
                                                   [3, 2,  1,  5],
                                                   [4, 7, -6,  1],
                                                   [6, 5,  3, -3]]))

        self.iteration = 0
        self.myPenalties = []
        self.enemyPenalties = []

        self.friend_seq = [1,2,1,0,3,2,1]
        self.realFriend_seq = []
        self.isFriend = True
        self.friendCounter = 0
        self.points = 2

    def setDecisions(self, myLastPenalty, currentDecision):
        self.myDecisions.append(currentDecision)


    def determineEnemyDecision(self, myLastPenalty):
        # print(self.myDecisions)
        myDecision = self.myDecisions[-1]
        if self.iteration != 0:
            self.enemyDecisions.append(np.where(self.myPenalty[myDecision] == myLastPenalty)[0][0])
            # print("cojest1", self.enemyPenalty)
            # print("cojest2", self.enemyPenalty[self.enemyDecisions[-1]])
            # print("cojest3", self.enemyPenalty[self.enemyDecisions[-1]][myDecision])
            # print("cojest4", myDecision)
            self.enemyPenalties.append(self.enemyPenalty[self.enemyDecisions[-1]][myDecision])
            self.myPenalties.append(myLastPenalty)




    def reset(self):
        if self.iteration % 101 == 0:
            self.enemyDecisions = [0]
            self.myDecisions = [0]
            self.iteration = 0
            self.myPenalties = []
            self.enemyPenalties = []
            self.isFriend = True
            self.realFriend_seq = []
            self.friendCounter = 0

    def iterating(self):
        self.iteration += 1
        # print(self.enemyDecisions)


    def prediction(self, d):
        return max(set(d), key=d.count)


    def last(self, lista):

        mostFreq = self.prediction(lista)

        hmm = None
        if len(lista) >= 3:
            if lista[-3] == lista[-2] and lista[-2] == lista[-1]:
                hmm = lista[-1]

        if hmm != None:
            choose = np.random.randint(0, 50)
            if choose < 20:
                return mostFreq
            else:
                return hmm
        else:
            return mostFreq


    def compare(self):
        # print(self.myPenalties)
        # print(self.enemyPenalties)
        if sum(self.myPenalties) > sum(self.enemyPenalties):
            return False
        else:
            return True


    def strategy(self):
        # if self.iteration == 100:           # jesli ostatnia iteracja
        #    if self.isFriend:               # i przeciwnik to buddy
        #        return 1                    # okaz mu milosc


        if self.iteration == 0:            # jesli poczatek walki
            return self.friend_seq[0]      # zagraj w gre
        elif self.friendCounter < len(self.friend_seq):     # jesli nie pierwsza iteracja i wskaznik mniejszy od dl listy
            # a przeciwnik to buddy i jego odpowiedz w poprzedniej rundzie byla przyjazna
            if self.isFriend and self.myPenalties[-1] == self.myPenalty[self.friend_seq[self.friendCounter]][self.friend_seq[self.friendCounter]]:
                self.friendCounter += 1                         # przesun wskaznik
                if self.friendCounter == len(self.friend_seq):
                    return 2
                else:
                    return self.friend_seq[self.friendCounter]    # zagraj w gre
            else:
                self.friendCounter += len(self.friend_seq)      # w przeciwnym razie zablokuj wskaznik
                self.isFriend = False                           # przygotuj sie do masakracji
        elif self.isFriend:                                                      # jesli wskaznik o odpowiedniej dlugosci
            if self.myPenalties[-1] == self.myPenalty[self.points][self.points]: # a odpowiedz przeciwnika wyborna
                return self.points                                               # zagraj w gre
            else:
                self.isFriend = False                                            # w przeciwnym razie szykuj sie do masakracji

        # MASAKRACJA
        lastDecisions = self.enemyDecisions[-10:]           # wez 10 poprzednich wynikow gracza
        pred = self.last(lastDecisions)               # wyznacz na ich podstawie jego nastepna decyzje

        # print(pred)
        myAdvantage = self.compare()

        if pred == 0:
            return 0

        elif pred == 1:
            return 1

        elif pred == 2:
            if myAdvantage:
                return 1
            else:
                dec = np.random.randint(0, 100)
                if dec <60:
                    return 2
                else:
                    return 1

        elif pred == 3:
            if myAdvantage:
                return 2
            else:
                return 3

        else:
            return 0


krzysio = Wiezien165173()


def decision(myLastPenalty):
    krzysio.determineEnemyDecision(myLastPenalty)

    currDecision = krzysio.strategy()

    krzysio.setDecisions(myLastPenalty, currDecision)
    krzysio.iterating()
    krzysio.reset()

    # print("\_(ツ)＿/")
    #print(currDecision)
    return currDecision