from tkinter import *
import os
class Sudoku:
    def __init__(self, Board):#Board  matrix of 9x9:
        try:
            for i in range(9):
                for j in range(9):
                    if int(Board[i][j]) >= 0 or int(Board[i][j]) <=9:
                        Board[i][j] = int(Board[i][j])
                    else:
                        raise ValueError("INCORRECT DATA")
        except:
            print('INITIALIZATION ERROR')
        self.__Board = Board
        self.__Solution = []

    def getBoard(self):
        return self.__Board
         

    def setSolution(self, Board):
        self.__Solution = Board

    def getNum(self, i,j):
        return self.__Board[i][j]

    def setNum(self,i,j,n):
        self.__Board[i][j] = n

    def getSolution(self):
        return self.__Solution


    def domainReduction(self, lin, col, updated_domain):

	#AC-3 Algorithm
        lin = int(lin)
        col = int(col)
        for c in range(0,9):#Checks if the number already exists on the line
            if self.getNum(lin, c) != 0:
                updated_domain.add(self.getNum(lin, c))
        
        for l in range(0,9):#Checks if the number already exists in the column
            if self.getNum(l, col) != 0:
                updated_domain.add(self.getNum(l, col))
        lr = int(lin/3)
        cr = int(col/3)
        for l in range(lr*3, (lr + 1)*3):
            for c in range(cr*3, (cr + 1)*3):
                if self.getNum(l, c) != 0:
                    updated_domain.add(self.getNum(l, c))
        return set(updated_domain)

    def resolve(self, i, j):
        row = i
        col = j
        flag = 0

        updatedDomain = set()
        oridomain = set([1, 2, 3, 4, 5, 6, 7, 8, 9])
        for row in range(9):
                 for col in range(9):
                         if self.getNum(row, col) == 0:
                                 flag = 1
                                 break
                 if flag == 1:
                         break
        if flag != 1 :
                 return 1
        else:
                 updatedDomain = self.domainReduction(row, col, updatedDomain)
                 updatedDomain = oridomain - updatedDomain
                 for value in set(updatedDomain):
                         self.setNum(row, col, value)
                         newSudoku = self.resolve(row, col)
                         if newSudoku == 1: 
                                 self.setSolution(self.__Board)
                                 self.writesSolution(self.getSolution())
                                 return 1
                         else: 
                                 self.setNum(row, col, 0)
                 self.setSolution(self.__Board)
                 self.writesSolution(self.getSolution())
                 return 0	

    def writesSolution(self, Solution):
        file1 = open("SudokuTEMP.txt", "w")
        try:
            for i in range (0,9):
                for j in range (0,9):
                    file1.write(str(Solution[i][j]))
                    file1.write(' ')
                file1.write('\n')
            file1.write('\n\n')
            file1.close()
        except:
            print("ERROR BY SAVING FILE")
        finally:
            file1.close()



class GUI:
    def __init__(self, toplevel):

        toplevel.resizable(width = False, height = False)
        toplevel.title('Sudoku')

        fonte = ('Arial', 18)

        self.frag = Frame(toplevel)
        self.frag.bind('<Motion>', self.corrects)
        self.frag.pack(ipady = 0, padx = 0)
        self.frag1 = Frame(toplevel)
        self.frag1.bind('<Motion>', self.corrects)
        self.frag1.pack(ipady = 0, padx = 0)
        self.frag2 = Frame(toplevel)
        self.frag2.bind('<Motion>', self.corrects)
        self.frag2.pack(ipady = 0, padx = 0)
        self.frag3 = Frame(toplevel)
        self.frag3.bind('<Motion>', self.corrects)
        self.frag3.pack(ipady = 0, padx = 0)
        self.frag4 = Frame(toplevel)
        self.frag4.bind('<Motion>', self.corrects)
        self.frag4.pack(ipady = 0, padx = 0)
        self.frag5 = Frame(toplevel)
        self.frag5.bind('<Motion>', self.corrects)
        self.frag5.pack(ipady = 0, padx = 0)
        self.frag6 = Frame(toplevel)
        self.frag6.bind('<Motion>', self.corrects)
        self.frag6.pack(ipady = 0, padx = 0)
        self.frag7 = Frame(toplevel)
        self.frag7.bind('<Motion>', self.corrects)
        self.frag7.pack(ipady = 0, padx = 0)
        self.frag8 = Frame(toplevel)
        self.frag8.bind('<Motion>', self.corrects)
        self.frag8.pack(ipady = 0, padx = 0)
        self.frag9 = Frame(toplevel)
        self.frag9.bind('<Motion>', self.corrects)
        self.frag9.pack(ipady = 1, padx = 1)

        self.__Board = []
        for i in range(1,10):
            self.__Board += [[0,0,0,0,0,0,0,0,0]]

        variable = self.frag
        px = 0
        py = 0
        cor = 'white'
        thickness = 0
        for i in range(0,9):
            for j in range(0,9):

                if i == 0:
                    variable = self.frag
                if i == 1:
                    variable = self.frag1
                if i == 2:
                    variable = self.frag2
                if i == 3:
                    variable = self.frag3
                if i == 4:
                    variable = self.frag4
                if i == 5:
                    variable = self.frag5
                if i == 6:
                    variable = self.frag6
                if i == 7:
                    variable = self.frag7
                if i == 8:
                    variable = self.frag8


                if j%2 == 0 and i%2 == 0:
                    thickness = 1
                if j%2 != 0 and i%2 != 0:
                    thickness = 1

                if j in [3,4,5] and i in [0,1,2,6,7,8]:
                    cor = 'gray'
                elif j not in [3,4,5] and i not in [0,1,2,6,7,8]:
                    cor = 'gray'
                else:
                    cor = 'white'
                self.__Board[i][j] = Entry(variable, width = 2, font = fonte, bg = cor, cursor = 'arrow', borderwidth = 0,
                                          highlightcolor = 'yellow', highlightthickness = 1, highlightbackground = 'black',
                                          textvar = graphicBoard[i][j])
                self.__Board[i][j].bind('<Button-1>', self.corrects)
                self.__Board[i][j].bind('<FocusIn>', self.corrects)
                self.__Board[i][j].bind('<Motion>', self.corrects)
                self.__Board[i][j].pack(side = LEFT, padx = px, pady = py)

                thickness = 0

        self.btn1 = Button(self.frag9, text = 'Save', fg = 'black', font = ('Arial', 13), command = self.solver)
        self.btn1.pack(side = RIGHT)

        self.btn2 = Button(self.frag9, text = 'Solve', fg = 'black', font = ('Arial', 13), command = self.open)
        self.btn2.pack(side = LEFT)

        self.btn3 = Button(self.frag9, text = 'Open', fg = 'black', font = ('Arial', 13), command = self.save)
        self.btn3.pack(side = LEFT)

        self.btn3 = Button(self.frag9, text = 'Reset', fg = 'black', font = ('Arial', 13), command = self.reset)
        self.btn3.pack(side = RIGHT)

        self.__states = "Input.txt"

    def open(self):
        try:
            Solution = Sudoku(self.getBoard())
            Solution.resolve(0,0)
            self.__states = "SudokuTEMP.txt"
            self.save()
            self.__states = "Input.txt"
            os.remove("SudokuTEMP.txt")
        except:
            print("READING MISTAKE")
        finally:
            self.__states = "Input.txt"


    def getBoard(self):
        Board = []
        for i in range(9):
            Board += [[0,0,0,0,0,0,0,0,0]]
        for i in range(9):
            for j in range(9):
                #self.__Board[i][j]
                Board[i][j] = graphicBoard[i][j].get()
                if Board[i][j] == '':
                    Board[i][j] = 0
        return Board

    def reset(self):
        for i in range(9):
            for j in range(9):
                graphicBoard[i][j].set('')

    def solver(self):
        f = open("Sudoku.txt", "a")
        try:
            for i in range (9):
                for j in range (9):
                    if self.__Board[i][j].get() == "":
                        f.write('0')
                    else:
                        f.write(self.__Board[i][j].get())
                    f.write(' ')
                f.write('\n')
            f.write('\n\n')
            f.close()
        except:
            print("ERROR BY SAVING FILE")
        finally:
            f.close()

    def corrects(self, event):
        for i in range(9):
            for j in range(9):
                if graphicBoard[i][j].get() == '':
                    continue
                if len(graphicBoard[i][j].get()) > 1 or graphicBoard[i][j].get() not in ['1','2','3','4','5','6','7','8','9']:
                    graphicBoard[i][j].set('')
    def complete(self):
        for i in range(0,9):
            for j in range(0,9):
                graphicBoard[i][j].set(self.__Board[i][j])

    def save(self):
        try:
            f = open(self.__states, 'r')

            text = f.readline()
            text = text.split(' ')
            for i in range(0,9):
                for j in range(0,9):
                    if text[0] == '0':
                        graphicBoard[i][j].set('')
                    else:
                        graphicBoard[i][j].set(text[0])
                    text.pop(0)
                text = f.readline()
                text = text.split(' ')
            f.close()

        except:
            print ("FATAL ERROR")
        finally:
            f.close()




Solution = []
tk = Tk()
txt = StringVar(tk)
graphicBoard = []
for i in range(1,10):
    graphicBoard += [[0,0,0,0,0,0,0,0,0]]
for i in range(0,9):
    for j in range(0,9):
        graphicBoard[i][j] = StringVar(tk)

a = GUI(tk)
tk.mainloop()
