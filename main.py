
class RowVector:
    def __init__(self, *args):
        pass

class ColumnVector:
    def __init__(self, *args):
        pass

class Matrix:
    def __init__(self, *args):
        pass
    def __getitem__(self, index):
        return
    
def add(a, b):
    return

def subtract(a, b):
    return

def scalar_multiply(mat, num):
    return

def multiply(a, b):
    return

def transfer_equations_to_matrix(equations, vals):
    return

def solve(mat):
    return


rows = []
if __name__ == '__main__':
    # below is just an example, you may change it to whatever you want, or use it to test your code
    with open('vector.csv', 'r', encoding='utf-8') as f:
        for i in f.readlines():
            line = ""  # here you need to split the line into a list of numbers, and make a RowVector
            print(line)
            # 
            rows.append()
        
    mat = Matrix(*rows)  # here you need to make a Matrix from the list of RowVectors
    print(mat)
    print(mat[0])
    print(len(mat))
    print(add(mat, mat))
    print(subtract(mat, mat))
    print(scalar_multiply(mat, 2))
    print(multiply(mat, mat))
    with open('equations.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        equations = []
        vals = []
        for i in lines:
            if i.beginwith('-'):
                opr = '-'
            else:
                opr = '+'
            for j in i:
                num = ''
                key = ''
                if i in ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'):
                    num += i
                    continue
                elif i == '=':
                    # add later into vals
                    break
                elif i in ('+', '-'):
                    opr = i
                    # add this into equations
                    continue
                else:
                    key += i
                    

    print(solve(transfer_equations_to_matrix(equations, vals)).out_latex_eqs())