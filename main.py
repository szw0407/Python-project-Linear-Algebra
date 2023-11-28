"""
This is the main file of Linear Algebra Project, an assignment for the Web Backend learners.

You need to implement the following classes and functions, according to the given `readme.md` file.

The task is a little bit challenging for beginners, wish you good luck!

If you have any questions, please contact us.

Notice:

- Pylance is set to strict mode, which means you need to follow the type hints strictly.
- Try your best to make your code Pythonic and elegant with Object-Oriented Programming.
- You can add more classes and functions if you want, just make sure the requirements are met.
"""


class Vector:
    """
    A basic Vector class

    Attributes:
        data (list[int | float | complex]): List containing the elements of the vector.

    Methods:
        __init__: Initializes the Vector object with the given elements.
        __getitem__: Returns the element at the specified index.
        __setitem__: Sets the element at the specified index to the given value.
        __len__: Returns the number of elements in the vector.
        __str__: Returns a string representation of the vector.
        __repr__: Returns a string representation of the vector that can be used to recreate the object.
        copy: Returns a copy of the vector.
        append: Appends a value to the end of the vector.
        pop: Removes and returns the last value of the vector.
        insert: Inserts a value at the specified index.
        remove: Removes and returns the value at the specified index.
    """
    data: list[int | float | complex]

    def __init__(self, *args: int | float | complex):
        self.data = list(args)

    def __getitem__(self, index: int):
        return self.data[index]

    def __setitem__(self, index: int, value: int | float | complex):
        self.data[index] = value

    def __len__(self):
        return len(self.data)

    def __str__(self) -> str:
        return ",".join([str(num) for num in self.data])

    def __repr__(self) -> str:
        return f"Vector({str(self)})"

    def copy(self):
        """
        Return a copy of the vector
        """
        return type(self)(*self.data)

    def append(self, value: int | float | complex):
        """
        Append a value to the end of the vector

        :param value: the value to append
        :return: None
        """
        self.data.append(value)

    def pop(self):
        """
        Pop the **last** value of the vector and return it

        :return: the last value
        """
        return self.data.pop()

    def insert(self, index:int, value:int|float|complex):
        """
        Insert a value to the vector at the given index

        :param index: the index where value is inserted
        :param value: the value to insert
        :return:  None
        """
        self.data.insert(index, value)

    def remove(self, index:int) -> int | float | complex:
        """
        Remove the value at the given index

        :param index: the index of the value to remove
        :return: the removed value
        """
        return self.data.pop(index)


class RowVector(Vector):
    """
    A basic RowVector class that inherits from Vector

    init method is not needed as it is the same as Vector:
    """
    def __str__(self) -> str:
        return "\t".join([str(i) for i in self.data])

    def __repr__(self) -> str:
        return f"RowVector({super().__str__()})"


class ColumnVector(Vector):
    """
    A basic ColumnVector class that inherits from Vector
    """
    def __str__(self) -> str:
        return "\n".join([str(i) for i in self.data])

    def __repr__(self) -> str:
        return f"ColumnVector({super().__str__()})"


class Matrix:
    """
    A basic Matrix class

    Args:
        *args (RowVector): Variable number of RowVector objects representing the rows of the matrix.

    Attributes:
        data (list[int|float|complex]): List containing the elements of the matrix.
        __row_count (int): Number of rows in the matrix.
        __column_count (int): Number of columns in the matrix.

    Methods:
        __init__: Initializes the Matrix object with the given rows.
        __getitem__: Returns the element at the specified index.
        __setitem__: Sets the element at the specified index to the given value.
        __len__: Returns the total number of elements in the matrix.
        count_rows: Returns the number of rows in the matrix.
        count_columns: Returns the number of columns in the matrix.
        get_columns: Returns a list of ColumnVector objects representing the specified columns.
        get_rows: Returns a list of RowVector objects representing the specified rows.
        __str__: Returns a string representation of the matrix.
        __repr__: Returns a string representation of the matrix that can be used to recreate the object.
        insert_row: Inserts a row at the specified index.
        insert_column: Inserts a column at the specified index.
        remove_row: Removes the row at the specified index and returns it.
        remove_column: Removes the column at the specified index and returns it.
        copy: Returns a copy of the matrix.
        pop_row: Removes and returns the last row of the matrix.
        pop_column: Removes and returns the last column of the matrix.
    """
    data: list[int|float|complex]
    __row_count: int = 0
    __column_count: int = 0

    def __init__(self, *args: RowVector):
        self.data = []
        for row in args:
            self.data.extend(row.data)
        self.__row_count = len(args)
        self.__column_count = len(args[0])

    def __getitem__(self, index:int) -> int | float | complex:
        return self.data[index]

    def __setitem__(self, index:int, value:int|float|complex):
        self.data[index] = value

    def __len__(self):
        return len(self.data)

    def count_rows(self):
        return self.__row_count

    def count_columns(self):
        return self.__column_count

    def get_columns(self, *args: int):
        """
        Return a list of ColumnVector, each ColumnVector is a column of the matrix

        :param args: the indices of the columns
        :return: a list of ColumnVector
        """
        return [ColumnVector(*self.data[i::self.__column_count]) for i in args]

    def get_rows(self, *args: int):
        """
        Return a list of RowVector, each RowVector is a row of the matrix

        :param args: the indices of the rows
        :return: a list of RowVector
        """
        return [RowVector(*self.data[i*self.__column_count:(i+1)*self.__column_count]) for i in args]

    def __str__(self) -> str:
        return "\n".join([str(row) for row in self.get_rows(*range(self.__row_count))])

    def __repr__(self) -> str:
        ret = ""
        for row in self.get_rows(*range(self.__row_count)):
            ret += ','.join([str(element) for element in row.data])
            ret = f'{ret[:-1]};'
        return f"Matrix({ret[:-1]})"

    def insert_row(self, index:int, row:RowVector):
        """
        Insert a row to the matrix at the given index

        :param index: the index where row is inserted
        :param row: the row to insert
        :return: None
        """
        left = self.data[:index*self.__column_count]
        right = self.data[index*self.__column_count:]
        self.data = left + row.data + right
        self.__row_count += 1

    def insert_column(self, index:int, column:ColumnVector):
        """
        Insert a column to the matrix at the given index

        :param index: the index where column is inserted
        :param column: the column to insert
        :return: None
        """
        column_generator = iter(column)
        for index in range(index, len(self.data) + len(column.data), self.__row_count):
            self.data.insert(index, next(column_generator))
        self.__column_count += 1

    def remove_row(self, index:int) -> RowVector:
        """
        Remove the row at the given index

        :param index: the index of the row to remove
        :return: the removed row
        """
        row = self.get_rows(index)
        left = self.data[:index*self.__column_count]
        right = self.data[(index+1)*self.__column_count:]
        self.data = left + right
        self.__row_count -= 1
        return row[0]

    def remove_column(self, index:int) -> ColumnVector:
        """
        Remove the column at the given index

        :param index: the index of the column to remove
        :return: the removed column
        """
        column = self.get_columns(index)[0]
        self.__column_count -= 1
        for ind in range(index, len(self) - len(column), self.__row_count):
            self.data.pop(ind)
        return column
    
    def copy(self):
        """
        Return a copy of the matrix
        """
        return type(self)(*self.get_rows(*range(self.__row_count)))

    def pop_row(self):
        """
        Pop the **last** row of the matrix and return it

        :return: the last row
        """
        self.__row_count -= 1
        ret = self.get_rows(self.__row_count)[0]
        self.data = self.data[:-self.__column_count]
        return ret

    def pop_column(self):
        """
        Pop the **last** column of the matrix and return it

        :return: the last column
        """
        self.__column_count -= 1
        ret = self.get_columns(self.__column_count)[0]
        for index in range(self.__column_count, len(self.data), self.__column_count):
            self.data.pop(index)
        return ret

    def append_row(self, row:RowVector):
        """
        Append a row to the end of the matrix

        :param row: the row to append
        :return: None
        """
        self.data.extend(row.data)
        self.__row_count += 1

    def append_column(self, column:ColumnVector):
        """
        Append a column to the end of the matrix

        :param column: the column to append
        :return: None
        """
        column_generator = iter(column)
        for index in range(self.__column_count, len(self.data) + len(column.data), self.__row_count):
            self.data.insert(index, next(column_generator))
        self.__column_count += 1
# def add(a, b):
#     return
#
#
# def subtract(a, b):
#     return
#
#
# def scalar_multiply(mat, num):
#     return
#
#
# def multiply(a, b):
#     return
#
#
# def transfer_equations_to_matrix(equations, vals):
#     return
#
#
# def solve(mat):
#     return


if __name__ == '__main__':
    # below is just an example, you may change it to whatever you want, or use it to test your code
    rows = []
    columns = []
    with open('vector.csv', 'r', encoding='utf-8') as f:
        for i in f:
            line = i.strip().split(',')
            # print(line)
            vec1 = RowVector(*map(eval, line))
            vec2 = ColumnVector(*map(eval, line))
            rows.append(vec1.copy())
            columns.append(vec2)
            print(f"A Row Vector:\n{vec1}")
            vec1.remove(-1)
            print(f"A Column Vector:\n{vec2}")
            vec2.append(0)

    mat = Matrix(*rows)  # here you need to make a Matrix from the list of RowVectors
    print(mat)
    print(mat[0])
    print(len(mat))
    print(mat.count_rows())
    print(mat.count_columns())
    print(mat.get_columns(0, 1))
    print(mat.get_rows(0, 1))
    print(f"to insert two rows:\n{rows[0]}\n{rows[1]}")
    mat.insert_row(0, rows[0])
    mat.append_row(rows[1])
    print(mat)
    print(f"to insert a column:\n{columns[0]}")
    mat.insert_column(0, columns[0])
    print(mat)
    mat.remove_row(0)
    print(mat.remove_column(0))
    print(mat)
    print(mat.copy())
    # print(add(mat, mat))
    # print(subtract(mat, mat))
    # print(scalar_multiply(mat, 2))
    # print(multiply(mat, mat))
    # with open('equations.txt', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     equations = []
    #     vals = []
    #     for i in lines:
    #         if i.beginwith('-'):
    #             opr = '-'
    #         else:
    #             opr = '+'
    #         for j in i:
    #             num = ''
    #             key = ''
    #             if i in ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'):
    #                 num += i
    #                 continue
    #             elif i == '=':
    #                 # add later into vals
    #                 break
    #             elif i in ('+', '-'):
    #                 opr = i
    #                 # add this into equations
    #                 continue
    #             else:
    #                 key += i

    # print(solve(transfer_equations_to_matrix(equations, vals)).out_latex_eqs())
