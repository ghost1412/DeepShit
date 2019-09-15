class utils:
    def num_there(self, val):
        return any(i.isdigit() for i in val)

    def windSet(self, data):
        tempVar = set()
        for line in range(0, len(data)):
            if self.num_there(data[line][-5]):
                tempVar.add(data[line][-6])
            else:
                tempVar.add(data[line][-5])
        tempVar = list(tempVar)
        tempVar.sort()
        return tempVar

    def weatherSet(self, data):
        tempVar = set()
        for line in range(0, len(data)):
            tempVar.add(data[line][-1])
            if data[line][-2] != " " and self.num_there(data[line][-2]) != 1:
                tempVar.add(data[line][-2])
        tempVar = list(tempVar)
        tempVar.sort()
        return tempVar

    def ntSameDataTime(self, *argv):
        if argv[0] == argv[1]:
            return 0
        else:
            return 1
